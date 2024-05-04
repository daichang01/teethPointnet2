import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()
#归一化点云，使用centroid为中心的坐标，求半径为1
def pc_normalize(pc):
    l = pc.shape[0]
    # 计算质心：首先计算点云所有点的均值（质心）。np.mean(pc, axis=0)计算每个维度上的均值，结果是一个形状为[D]的向量。
    centroid = np.mean(pc, axis=0)
    # 去中心化：接下来，从每个点减去质心，使点云的中心移至原点。这一步是去中心化，有助于消除数据在空间上的偏移。
    pc = pc - centroid
    #缩放处理：计算去中心化后每个点到原点的欧氏距离，然后找到这些距离中的最大值。
    # 这里，np.sqrt(np.sum(pc**2, axis=1))计算的是每个点到原点的距离，np.max则找出这些距离中的最大值。
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    # 归一化缩放：最后，将点云中的每个点除以最大距离m，确保所有点都在单位球内。这一步使得点云的尺寸统一，有利于后续处理的稳定性和性能。
    pc = pc / m
    return pc

# square_distance函数用来在ball query过程中确定每一个点距离采样点的距离
# 函数输入是两组点，N为第一组点src的个数，M为第二组点dst的个数，C为输入点的通道数（如果是xyz时是C = 3
# 函数返回的是两组点两两之间的欧几里德举例，即NxM矩阵
# 在训练中数据以mini-batch的形式输入，所以一个batch数量的维度为B

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

#按照输入的点云数据和索引返回索引的点云数据
# 例如points为Bx2048x3点云，idx为[5666, 1000, 2000]
# 则返回Batch中第5666, 1000, 2000个点组成的B✖4✖3的点云集
# 如果idx为一个[B, D1, ...DN],则他会按照idx中的维度结构将其提取为[B, D1, ...DN，C]
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

#farthest_point_sample函数完成最远点采样
# 从一个输入点云中按照所需要的点的个数npoint采样出足够多的点
# 并且点与点之间的距离要足够远
# 返回结果是npoint个采样点在原始点云中的索引
def farthest_point_sample(xyz, npoint):
    """
    远点采样(Farthest Point Sampling, FPS)算法用于点云下采样。
    它迭代地选择最远的点，这些点是从已选择的点集合中最远的。

    输入:
        xyz: 点云数据，形状为[B, N, 3]，其中B是批次大小，N是点的数量，3代表x, y, z坐标。
        npoint: 要采样的点数。
    返回:
        centroids: 采样点的索引，形状为[B, npoint]。
    """
    # 获取输入点云的设备（CPU/GPU），以确保所有计算都在同一设备上执行。
    device = xyz.device
    
    # 初始化变量
    B, N, _ = xyz.shape  # 批次大小，点的数量，和坐标维度
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 将采样点索引centroids初始化为全0张量，形状为[B, npoint]。
    distance = torch.ones(B, N).to(device) * 1e10  # 初始化一个距离张量distance，形状为[B, N]，所有值初始化为一个很大的数（1e10），表示初始时所有点都非常远。
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 通过torch.randint随机选择每批的第一个采样点的索引farthest。
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # 生成批次索引
    # 直到采样点达到npoint，否则进行如下迭代
    for i in range(npoint):
        centroids[:, i] = farthest  # 在每次迭代中，首先将当前最远点的索引保存到centroids中。
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 根据当前最远点的索引farthest，提取对应的点坐标centroid，形状为[B, 1, 3]
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 计算所有点到centroid的平方距离，并更新到distance中。
        mask = dist < distance  # 找到距离小于当前最远距离的点
        distance[mask] = dist[mask]  # 更新这些点的最远距离
        farthest = torch.max(distance, -1)[1]  # 在更新完所有点的最远距离后，从distance中找到新的最远点，即下一个采样点。
    
    return centroids  # 返回采样点索引

# query_ball_point函数用于寻找球形领域中的点
# 输入中radius为球形领域的半径，nsample为每个领域中要采样的点
# new_xyz为centroids点的数据，xyz为所有的点云数据
# 输出为每个样本的每个球形领域的nsample个采样点集的索引[B, S, nsample]

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    #sqrdists：[B,S,N]记录S个中心点（new_xyz)与所有点（xyz）之间的欧几里德距离
    sqrdists = square_distance(new_xyz, xyz)
    #找到所有举例大于radius**2的点，其group_idx直接置为N，其余的保留原来的值
    group_idx[sqrdists > radius ** 2] = N
    # 对每个查询点的索引进行排序，然后取前nsample个索引。如果某个查询点的局部区域内的点数少于nsample，那么这个区域的点索引将被重复选取，直到达到nsample个。
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# sampling+grouping主要用于将整个点云分散成局部的group
# 对每一个group都可以用pointnet单独的提取局部的全局特征。
# sampling+狗肉平分成了sample_and_group和sample_and_group_all 两个函数
# 其区别在于sample_and_group_all直接将所有点作为一个group
# 例如
# 512 = npoint:points sampled in farthest point sampling
# 0.2 = radius:search radius in local region
# 32 = nsample :how mant points in each local region
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

#sample_and_group_all直接将所有点作为一个group；npoint =1
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# PointNetSetAbstraction类实现普通的set abstraction:
# 首先通过sample_and_group的操作形成局部group
# 然后对局部group的每一个点做MLP操作，最后进行局部的最大出化，得到局部的全局特征
class PointNetSetAbstraction(nn.Module):
#     类初始化方法 __init__
# npoint: 抽样点的数量，即在每次迭代中要从输入点云中抽取的关键点数。
# radius: 在局部区域内搜索邻近点时使用的搜索半径。
# nsample: 每个局部区域内要采样的点的数量。
# in_channel: 输入特征的通道数。
# mlp: 一个列表，定义了一系列卷积层的输出通道数。这些卷积层用于处理每个局部区域内的点云数据，从而学习更深层次的特征表示。
# group_all: 一个布尔值，指示是否对所有点应用分组操作。如果为True，则不进行抽样，而是使用所有点。
# 类中使用了nn.ModuleList()来保存卷积层(mlp_convs)和批归一化层(mlp_bns)。这些层将按照mlp列表中定义的通道数依次应用于每个局部区域的点云数据。
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
# 前向传播方法 forward
# 输入参数xyz是点的位置数据，形状为[B, C, N]，其中B是批次大小，C是坐标维度(通常为3，对应于X、Y、Z)，N是每个批次中点的数量。
# 输入参数points是点的特征数据，形状为[B, D, N]，其中D是特征维度。
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# **sampling**采样：选取centroid（中心点）   (sample centroids)

# **grouping**分组：以centroid为中心，选取局部的点  (group points by centroids)

# **PointNet**：对分组内的点应用pointnet进行特征的学习 (apply PointNet on each point group)

# 以上过程加起来称作**Set Abstraction**
        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])

class PointNetSetAbstractionMsg(nn.Module):
    # 构造函数
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        # 保存构造函数参数
        self.npoint = npoint  # 需要采样的点的数量
        self.radius_list = radius_list  # 搜索半径的列表
        self.nsample_list = nsample_list  # 每个半径内的点的数量
        self.conv_blocks = nn.ModuleList()  # 卷积块列表
        self.bn_blocks = nn.ModuleList()  # 批归一化块列表

        # 初始化多层感知机（MLP）模块
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()  # 当前半径下的卷积层列表
            bns = nn.ModuleList()  # 当前半径下的批归一化层列表
            last_channel = in_channel + 3  # 初始通道数，点特征加上空间坐标
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))  # 添加卷积层
                bns.append(nn.BatchNorm2d(out_channel))  # 添加批归一化层
                last_channel = out_channel  # 更新通道数
            self.conv_blocks.append(convs)  # 添加到总的卷积块列表
            self.bn_blocks.append(bns)  # 添加到总的批归一化块列表

    # 前向传播函数
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]  # 输入点的位置数据
            points: input points data, [B, D, N]  # 输入点的特征数据
        Return:
            new_xyz: sampled points position data, [B, C, S]  # 采样点的位置数据
            new_points_concat: sample points feature data, [B, D', S]  # 采样点的特征数据
        """
        xyz = xyz.permute(0, 2, 1)  # 交换维度，为了索引操作
        if points is not None:
            points = points.permute(0, 2, 1)  # 如果有特征数据，也交换维度

        B, N, C = xyz.shape  # 获取批量大小、点数和坐标维度
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  # 采样最远点
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  # 根据半径和点数获取邻域索引
            grouped_xyz = index_points(xyz, group_idx)  # 获取邻域点的坐标
            grouped_xyz -= new_xyz.view(B, S, 1, C)  # 中心化邻域点
            if points is not None:
                grouped_points = index_points(points, group_idx)  # 获取邻域点的特征
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # 合并特征和坐标
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # 调整维度以适应卷积操作
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]  # 获取卷积层
                bn = self.bn_blocks[i][j]  # 获取批归一化层
                grouped_points = F.relu(bn(conv(grouped_points)))  # 应用卷积、批归一化和ReLU激活
            new_points = torch.max(grouped_points, 2)[0]  # 应用最大池化
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)  # 调整采样点坐标的维度
        new_points_concat = torch.cat(new_points_list, dim=1)  # 合并所有特征
        return new_xyz, new_points_concat


#feature propagation实现主要通过线性插值和MLP完成
    # 当点的个数只有一个的时候，采用repeat直接复制成N个点
    # 当点的个数大于一个的时候，采用线性差值的方式进行上采样
    # 拼接上下采样对应点的SA层的特征，再对拼接后的每一个点做一个MLP
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            #当点的个数只有一个的时候，采用repeat直接复制成N个点
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 当点的个数大于一个的时候，采用线性差值的方式进行上采样
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8) #距离越远的点权重越小
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm #对于每一个点的权重再做一个全局的归一化
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

