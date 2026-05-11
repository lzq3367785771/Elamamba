import os
import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, balanced_accuracy_score

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
    ]
)

train_transforms_raw = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)

    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    print_log(f"{'=' * 50}", logger=logger)
    print_log(f"【ElaMamba Architecture Core Metrics】", logger=logger)
    total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print_log(f"Param. (M): {total_params / 1e6:.2f} M", logger=logger)
    print_log(f"{'=' * 50}", logger=logger)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
        
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    base_model.zero_grad()
    misc.summary_parameters(base_model, logger=logger)

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()

        n_batches = len(train_dataloader)
        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            
            if 'scan' in args.config:
                points = train_transforms(points)
            else:
                points = train_transforms_raw(points)

            model_outputs = base_model(points)

            if isinstance(model_outputs, tuple):
                if len(model_outputs) == 3:
                    ret, concat_f, offsets = model_outputs
                elif len(model_outputs) == 2:
                    ret, concat_f = model_outputs
                    offsets = None
                else:
                    ret = model_outputs[0]
                    offsets = None
            else:
                ret = model_outputs
                offsets = None

            model_for_loss = base_model.module if hasattr(base_model, 'module') else base_model
            loss, acc = model_for_loss.get_loss_acc(ret, label)

            if offsets is not None:
                l2_lambda = 0.001
                offset_loss = torch.mean(offsets ** 2)
                loss = loss + l2_lambda * offset_loss

            _loss = loss
            _loss.backward()

            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            better = metrics.better_than(best_metrics)
            if better and ('fewshot' not in args.config):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log("-" * 92, logger=logger)
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config,
                                                 logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log("*" * 88, logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote,
                                                'ckpt-best_vote', args, logger=logger)
        if 'fewshot' not in args.config:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args,
                                    logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    base_model.eval()
    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            points = misc.fps(points, npoints)
            logits = base_model(points)
            if isinstance(logits, tuple): logits = logits[0]
            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)
            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  OA = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)

def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger=logger)
    base_model.eval()
    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()
                points = test_transforms(points)
                logits = base_model(points)
                if isinstance(logits, tuple): logits = logits[0]
                target = label.view(-1)
                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)
            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        raise NotImplementedError()
    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()
    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        inference_time_list = []
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if idx <= 0.2 * len(test_dataloader):
                points = data[0].cuda()
                label = data[1].cuda()
                points = misc.fps(points, npoints)
                logits = base_model(points)
                if isinstance(logits, tuple): logits = logits[0]
                target = label.view(-1)
                pred = logits.argmax(-1).view(-1)
                test_pred.append(pred.detach())
                test_label.append(target.detach())
            else:
                torch.cuda.synchronize()
                time_start = time.time()
                points = data[0].cuda()
                label = data[1].cuda()
                points = misc.fps(points, npoints)
                logits = base_model(points)
                if isinstance(logits, tuple): logits = logits[0]
                target = label.view(-1)
                pred = logits.argmax(-1).view(-1)
                test_pred.append(pred.detach())
                test_label.append(target.detach())
                torch.cuda.synchronize()
                inference_time_list.append(time.time() - time_start)

            # import glob
            # save_dir = "/root/autodl-tmp/PointMamba-official_backup_20260411/PointMamba_v3/Paper_Vis_Data/ScanObject_ElaMamba_gold"
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir, exist_ok=True)
            # pm_wrong_dir = "/root/autodl-tmp/PointMamba-official_backup_20260411/PointMamba_v3/Paper_Vis_Data/ScanObject_PointMamba_false"
            # if not hasattr(base_model, 'target_ids'):
            #     base_model.target_ids = set()
            #     for f in glob.glob(f"{pm_wrong_dir}/*.npy"):
            #         name = os.path.basename(f).replace('.npy', '')
            #         parts = name.split('_')
            #         base_model.target_ids.add(f"{parts[-2]}_{parts[-1]}")
            #     from collections import defaultdict
            #     base_model.class_golden_counts = defaultdict(int)

            # for i in range(points.shape[0]):
            #     pts_array = points[i].cpu().numpy()
            #     gt_lbl = target[i].cpu().item()
            #     pred_lbl = pred[i].cpu().item()
            #     identifier = f"b{idx}_i{i}"
            #     if identifier in base_model.target_ids and gt_lbl == pred_lbl:
            #         if base_model.class_golden_counts[gt_lbl] < 3:
            #             save_name = f"{save_dir}/Golden_GT_{gt_lbl}_ElaPred_{pred_lbl}_{identifier}.npy"
            #             np.save(save_name, pts_array)
            #             base_model.class_golden_counts[gt_lbl] += 1
            #             print(f"🌟 捕获 [类别 {gt_lbl}] 黄金样本 ({base_model.class_golden_counts[gt_lbl]}/3): {save_name}")

        inference_time = np.mean(inference_time_list)
        inference_fps = 1 / inference_time * config.total_bs
        print_log(f"[TEST] inference time: {inference_time}", logger=logger)
        print_log(f"[TEST] inference fps: {inference_fps}", logger=logger)
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] Overall Accuracy (OA) = %.4f' % acc, logger=logger)

        if args.vote:
            if args.distributed:
                torch.cuda.synchronize()
            print_log(f"[TEST_VOTE]", logger=logger)
            best_vote_acc = 0.
            for vote_step in range(1, 300):
                this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
                if best_vote_acc < this_acc:
                    best_vote_acc = this_acc
                print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (vote_step, this_acc, best_vote_acc),
                          logger=logger)
            print_log('[TEST_VOTE] acc = %.4f' % best_vote_acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    base_model.eval()
    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)
            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)
            local_pred = []
            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()
                points = test_transforms(points)
                logits = base_model(points)
                if isinstance(logits, tuple): logits = logits[0]
                target = label.view(-1)
                local_pred.append(logits.detach().unsqueeze(0))
            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)
            test_pred.append(pred_choice)
            test_label.append(target.detach())
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        if args.distributed:
            torch.cuda.synchronize()
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    return acc

def plot_embedding(data, label, title, category_nums):
    TSNE_PATH = "./vis/tsne/"
    colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990',
              '#dcbeff', '#9A6324', '#800000', '#000075', '#a9a9a9', '#888870', '#000000'
              ]
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(8, 8))
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], s=8, marker='o', c=colors[int(label[i])], cmap='coolwarm')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.axis('off')
    if not os.path.isdir(TSNE_PATH):
        os.makedirs(TSNE_PATH)
    plt.savefig(TSNE_PATH + "tsne_fix_.pdf")
    return fig

def test_only_tsne(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()
    test_label = []
    test_feature = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            points = misc.fps(points, npoints)
            _, concat_f = base_model(points)
            target = label.view(-1)
            test_label.append(target.detach())
            test_feature.append(concat_f.detach())
        test_label = torch.cat(test_label, dim=0)
        category_nums = config.model.cls_dim
        index = test_label < category_nums
        label_all = test_label[index]
        test_feature = torch.cat(test_feature, dim=0)
        test_feature = test_feature[index]
        test_feature = test_feature.cpu().numpy()
        label = label_all.cpu().numpy()
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(test_feature.squeeze())
        fig = plot_embedding(result, label, '', category_nums)

def test_tsne(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        raise NotImplementedError()
    test_only_tsne(base_model, test_dataloader, args, config, logger=logger)