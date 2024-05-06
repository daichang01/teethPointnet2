# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

# ShapeNet用来训练部件分割（part segmentation)
# 训练集有14007个点云，测试集有2874个点云

#点云归一化，以cnetroid为中心，半径为1
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0) # 得到质心坐标
    pc = pc - centroid # 通过从每个点的坐标中减去质心坐标，将点云的中心移动到原点。这一步是为了去除点云位置的影响，仅保留形状信息。
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) #找出所有点中到原点距离最远的点的距离，即点云的最大范围。
    pc = pc / m #将点云中的每个点除以最大距离 m，确保点云中所有点的最大距离为1。这样，点云被规范到单位球内，使得不同的点云数据具有可比性
    return pc
# 指定数据集在哪个目录下，默认采2500个点
class PartNormalDataset(Dataset):
    def __init__(self,root = 'data/teeth_segmentation_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'train_files.json'), 'r') as f:
            train_ids = set([str(d.split('/')[3]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'val_files.json'), 'r') as f:
            val_ids = set([str(d.split('/')[3]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'test_files.json'), 'r') as f:
            test_ids = set([str(d.split('/')[3]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        # 1个物体，3个部件
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Teeth': [0, 1, 2]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    #获取采样点数
    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        #随机选择npoints个点
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        #point_set 点云数据，前6列，进行了点云归1化，限制半径为1  (2048, 6)
        #cls 类别，这里只有一个类别0,代表teeth   0
        #seg 最后一列标签，是向量，代表每个点属于哪个类别 (2048,)
        return point_set, cls, seg #返回一个索引对应的数据项 

    def __len__(self):
        return len(self.datapath)

def main():
    root = 'data/teeth_segmentation_normal'

    TRAIN_DATASET = PartNormalDataset(root, npoints=1800, split='trainval', normal_channel=False)

if __name__ == '__main__':
    main()