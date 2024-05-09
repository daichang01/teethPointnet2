"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.teethSegDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#字典的定义
seg_classes = {'Teeth': [0, 1, 2]}

seg_label_to_cat = {}  # {0:Teeth}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def visualize_point_cloud(points, labels, num_classes):
    """Visualize a point cloud with colors assigned to each point based on the segmentation labels."""
    # 确保点的形状和数据类型是正确的
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点数组应该是 (N, 3) 形状")
    if points.dtype not in [np.float32, np.float64]:
        points = points.astype(np.float32)  # 转换为 float32 类型

    # 创建颜色映射
    color_map = np.array([
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        # 根据需要添加更多颜色
    ])
    if num_classes > len(color_map):
        # 如果类别数大于颜色映射数组的长度，随机生成额外的颜色
        extra_colors = np.random.rand(num_classes - len(color_map), 3)
        color_map = np.vstack([color_map, extra_colors])

    colors = color_map[labels % num_classes]  # 使用模运算确保标签索引不会超出颜色映射的范围

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))  # 确保颜色也是正确的数据类型
    o3d.visualization.draw_geometries([pcd])

# 使用示例：
# visualize_point_cloud(points_np[0], pred_labels_np[0], num_classes)

def save_point_cloud_txt(points, labels, file_path):
    """Save points and labels to a TXT file."""
    # 将点坐标和标签合并
    data_to_save = np.hstack((points, labels[:, np.newaxis]))  # 确保标签是列向量
    # 保存数据到文件
    np.savetxt(file_path, data_to_save, fmt='%f %f %f %d')  # XYZ为浮点数，标签为整数

def restore_original_scale(pc_normalized, centroid, max_dist):
    return pc_normalized * max_dist + centroid
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    #数据位置
    root = 'data/teeth_segmentation_normal/'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 1
    num_part = 3

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    #测试不需要权重更新
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)] #list:3 元素初始化为0
        total_correct_class = [0 for _ in range(num_part)] #list: 3 元素初始化为0
        shape_ious = {cat: [] for cat in seg_classes.keys()} # dict： 1个类别
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size() #cur_batch_size:8, num_point:2048
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            #对每个点进行分类
            target = target.cpu().data.numpy() #(8, 2048)

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]] #类别字符串：如‘airplane
                logits = cur_pred_val_logits[i, :, :] #（2048, 3）
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target) #正确分类的点数  (8,2048)
            total_correct += correct #累计正确分类的点数
            total_seen += (cur_batch_size * NUM_POINT) #累计测试的点数

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :] #(2048)
                segl = target[i, :] #(2048)
                cat = seg_label_to_cat[segl[0]] #‘Teeth
                # 计算part IoU
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))] #list:3
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l))) #计算交并比
                # 计算类别的shape IoU
                shape_ious[cat].append(np.mean(part_ious)) # dict:16个类别（我这里是一个类别，应该时一样的）
            
            seg_pred = seg_pred.argmax(dim=2)  # 获取最可能的分类标签
            # 将数据从GPU移动到CPU，并转换为numpy数组
            points_np = points.cpu().numpy()
            pred_labels_np = seg_pred.cpu().numpy()

            # 提取每个样本的 XYZ 坐标，假设前三个特征是 XYZ
            points_xyz = points_np[:, :3, :]  # 修改为正确的维度切片，形状应为 (8, 3, 2048)
            points_xyz = points_xyz.transpose(0, 2, 1)  # 调整形状为 (8, 2048, 3) 以符合 open3d 的期望

             # 保存每个样本的点云到 TXT 文件
            for i in range(points_xyz.shape[0]):
                file_path = os.path.join('pred_point_clouds', f'point_cloud_batch{batch_id}_sample{i}.txt')
                save_point_cloud_txt(points_xyz[i], pred_labels_np[i], file_path)
            
            # 可视化第一个批次的第一个点云样本
            if batch_id == 0:  # 仅可视化第一个批次的第一个点云
                visualize_point_cloud(points_xyz[0], pred_labels_np[0], num_part)




        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
