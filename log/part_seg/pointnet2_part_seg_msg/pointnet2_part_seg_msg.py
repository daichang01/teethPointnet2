import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        # 判断是否包含法线或其他附加通道
        if normal_channel:
            additional_channel = 3  # 如果包含法线通道，则添加三个附加通道
        else:
            additional_channel = 0  # 否则不添加附加通道
        self.normal_channel = normal_channel  # 保存是否包含附加通道的信息
        
        # 定义第一层的点集抽象模块，配置多尺度抽象的参数
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # 定义第二层的点集抽象模块
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        # 定义第三层的点集抽象模块，这是全局特征抽象层
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # 定义特征传播模块，用于特征向较高分辨率层级的传播
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=135+additional_channel, mlp=[128, 128])
        
        # 定义一维卷积层，用于处理传播后的特征
        self.conv1 = nn.Conv1d(128, 128, 1)
        # 定义批量归一化层
        self.bn1 = nn.BatchNorm1d(128)
        # 定义Dropout层，用于防止过拟合
        self.drop1 = nn.Dropout(0.5)
        # 定义最终的一维卷积层，输出层，用于分类，输出每个类的得分
        self.conv2 = nn.Conv1d(128, num_classes, 1)


    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape  # 提取批次大小B，特征数量C，和每批次点数N
        if self.normal_channel:
            l0_points = xyz  # 如果有额外的通道信息，保留原始点云数据
            l0_xyz = xyz[:,:3,:]  # 只取前三个通道的数据，通常是XYZ坐标
        else:
            l0_points = xyz  # 否则不区分额外通道，直接使用原始数据
            l0_xyz = xyz  # 点坐标与点特征数据相同
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # 第一层set abstraction操作
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # 第二层set abstraction操作
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # 第三层set abstraction操作
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # 从l3到l2的特征传播
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # 从l2到l1的特征传播
        cls_label_one_hot = cls_label.view(B,1,1).repeat(1,1,N)  # 将类标签转换为one-hot编码，并重复以匹配每个点
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)  # 从l1到l0的特征传播，包括类别信息
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))  # 对最初层点的特征应用卷积、批归一化和ReLU激活函数
        x = self.drop1(feat)  # 应用Dropout
        x = self.conv2(x)  # 应用第二个卷积层
        x = F.log_softmax(x, dim=1)  # 使用对数Softmax获取预测的对数概率
        x = x.permute(0, 2, 1)  # 调整输出维度顺序以符合期望的输出格式
        return x, l3_points  # 返回最终的分类结果和第三层点的特征



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss