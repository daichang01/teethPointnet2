import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
# PointNetSetAbstractionMsg (Multi-scale Grouping Layer)
# self.sa1
# npoint=512: 在第一层特征抽象中，随机采样512个点。
# radius=[0.1, 0.2, 0.4]: 使用不同的搜索半径来捕获多尺度的局部结构。这意味着在每个半径级别，都会基于中心点收集周围的点。
# nsample=[32, 64, 128]: 分别对应上述每个半径的采样点数，定义了在每个搜索半径下，围绕中心点采样的邻居点数量。
# in_channel=3+additional_channel: 输入通道数，通常对应点云的XYZ坐标加上可能的额外通道（如颜色、强度等）。
# mlp=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]: 这是一个多层感知机（MLP）的列表，每个列表项对应一个半径级别的MLP配置，用于处理从每个邻域提取的特征。
# self.sa2
# npoint=128: 在第二层特征抽象中，从前一层的输出中进一步随机采样128个点。
# radius=[0.4, 0.8]: 增加搜索半径以捕获更广的局部结构。
# nsample=[64, 128]: 对应上述每个半径的采样点数。
# in_channel=128+128+64: 输入通道数，这是从第一层特征抽象中传递下来的特征维数之和。
# mlp=[[128, 128, 256], [128, 196, 256]]: 为这一层的每个半径配置的MLP。
# PointNetSetAbstraction (Global Feature Layer)
# self.sa3
# npoint=None, radius=None, nsample=None: 这些参数为空或None，表示这一层处理所有点，也就是进行全局特征抽象。
# in_channel=512 + 3: 输入通道数，这是从上一层传递下来的特征维数加上原始的点坐标维数。
# mlp=[256, 512, 1024]: 用于处理全局特征的MLP配置。
# group_all=True: 这个参数指示该层要抽取全局特征，不再进行局部采样。
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss