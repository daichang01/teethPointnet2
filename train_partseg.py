"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_utils.teethSegDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Teeth': [0, 1, 2]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number') #点云数据是2048个点
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/teeth_segmentation_normal/'

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # 设定类别总数和部分的数量
    num_classes = 16
    num_part = 3

    '''MODEL LOADING'''
    # 导入模型定义模块，假设 args.model 中包含了模型的模块名
    MODEL = importlib.import_module(args.model)
    # 将模型定义文件复制到实验目录，以保存代码版本
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # 将辅助工具文件也复制到实验目录
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    # 创建分类器模型实例，并将其转移到CUDA设备上
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    # 获取损失函数，并将其转移到CUDA设备上
    criterion = MODEL.get_loss().cuda()
    # 应用ReLU激活函数的原地操作优化
    classifier.apply(inplace_relu)

    # 定义权重初始化函数
    def weights_init(m):  # 权重初始化
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            # 对卷积层使用Xavier初始化方法
            torch.nn.init.xavier_normal_(m.weight.data)
            # 将偏置初始化为0
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            # 对全连接层使用Xavier初始化方法
            torch.nn.init.xavier_normal_(m.weight.data)
            # 将偏置初始化为0
            torch.nn.init.constant_(m.bias.data, 0.0)

    # 尝试从检查点加载模型
    try:
        # 加载最优模型的检查点文件
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        # 获取保存的训练轮数
        start_epoch = checkpoint['epoch']
        # 加载模型状态
        classifier.load_state_dict(checkpoint['model_state_dict'])
        # 日志输出使用预训练模型
        log_string('Use pretrain model')
    except:
        # 没有找到预训练模型时，从头开始训练
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        # 应用权重初始化
        classifier = classifier.apply(weights_init)

    # 根据参数选择优化器
    if args.optimizer == 'Adam':  # 使用Adam优化器
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        # 默认使用SGD优化器
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)


    # 目的是调整PyTorch模型中所有批归一化（Batch Normalization，BN）层的动量（momentum）参数。
    # 这个函数特别适用于深度学习训练过程中，当你希望动态调整BN层的动量以优化模型训练性能时
    def bn_momentum_adjust(m, momentum):
        # 如果是 BatchNorm2d 或 BatchNorm1d 层，调整动量
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    # 定义学习率的下限
    LEARNING_RATE_CLIP = 1e-5
    # 原始的动量值
    MOMENTUM_ORIGINAL = 0.1
    # 动量的衰减率
    MOMENTUM_DECCAY = 0.5
    # 动量的衰减步长
    MOMENTUM_DECCAY_STEP = args.step_size

    # 记录最佳准确率、全局周期数和最佳类别平均IoU和实例平均IoU
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    # 开始训练循环
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        # 调整学习率和 BatchNorm 层的动量
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 动量的衰减
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        # 更新 BatchNorm 层的动量
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        # 设置模型为训练模式
        classifier = classifier.train()

        '''learning one epoch'''
        # 每个epoch学习一次
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            print(len(points))
            print(label)
            print(len(target))
            points = points.data.numpy()
            # 数据增强
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            # 计算输出
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            # 计算准确率
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            # 计算损失
            loss = criterion(seg_pred, target, trans_feat)
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)


        # 性能评估
        # 关闭梯度计算，节省内存和计算资源，因为在测试阶段不需要进行反向传播
        with torch.no_grad():
            test_metrics = {}  # 存储测试指标的字典
            total_correct = 0  # 总正确数
            total_seen = 0  # 总样本数
            total_seen_class = [0 for _ in range(num_part)]  # 每个类别的总样本数
            total_correct_class = [0 for _ in range(num_part)]  # 每个类别的正确样本数
            shape_ious = {cat: [] for cat in seg_classes.keys()}  # 存储每个类别的IoU
            seg_label_to_cat = {}  # 分割标签到类别的映射

            # 建立分割标签到类别的映射
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            # 设置模型为评估模式
            classifier = classifier.eval()

            # 迭代测试数据集
            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                # 计算当前批次的预测准确性
                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                # 计算每个类别的准确性
                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                # 计算IoU，用于评估分割质量
                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            # 计算所有形状的平均IoU
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

            log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
            if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_metrics['accuracy'],
                    'class_avg_iou': test_metrics['class_avg_iou'],
                    'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)  # 保存模型
                log_string('Saving model....')

            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
            if test_metrics['class_avg_iou'] > best_class_avg_iou:
                best_class_avg_iou = test_metrics['class_avg_iou']
            if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
                best_inctance_avg_iou = test_metrics['inctance_avg_iou']
            log_string('Best accuracy is: %.5f' % best_acc)
            log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
            log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
            global_epoch += 1  # 更新全局周期数



if __name__ == '__main__':
    args = parse_args()
    main(args)
