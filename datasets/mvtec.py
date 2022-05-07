import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T    #引入几个用于加载数据集的库


#这是数据集中所有种类的数据
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    # 对数据集进行初始化，设定了一些默认参数
    def __init__(self, dataset_path= 'D:\\datasets\\mvtec_anomaly_detection', class_name='bottle', is_train=True,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # 如果不存在，则进行下载
        # self.download()

        # load dataset，这个x和y对应的是
        self.x, self.y, self.mask = self.load_dataset_folder()

        # 这是对数据进行transform
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])]) #对这个进行正则化，方便模型的训练
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        #该数据集中处理较好，已经将所有缺陷的mask都制作出来了
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')  #转换成传统的RGB三通道图像
        x = self.transform_x(x) #利用自定义函数里面的函数进行数据预处理

        if y == 0: #y=0代表的是如果没有标签，则代表是正常的产品，没有缺陷。
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:      #如果有标签代表该图像是缺陷图像
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask


    def __len__(self):
        return len(self.x)

    #加载数据在这里
    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase) #这是图片路径，相对应的train或者test
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth') #这是mask路径

        img_types = sorted(os.listdir(img_dir))  #对数据集中的文件夹进行排序，保证程序每次暂存的结果都是稳定的，主要是针对test数据集
        for img_type in img_types:
            # 加载具体图片数据
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):  #保险起见再对其是否是dir进行判断，则直接开始下次循环
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')]) #获取png格式的文件路径进行获取
            x.extend(img_fpath_list)  #将数据扩展到列表当中去，x存储的是所原始图像的路径，不是源文件

            # load gt labels，加载mask标签
            if img_type == 'good': #代表该图片是正常图片，将对应的y标签设置为0，并且将掩膜数据扩展整个文件夹长度
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type) #找到掩膜中相对应的文件夹文件位置
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list) #将对应的mask路径加载进来

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

#     def download(self):
#         """Download dataset if not exist"""

#         if not os.path.exists(self.mvtec_folder_path):
#             tar_file_path = self.mvtec_folder_path + '.tar.xz'
#             if not os.path.exists(tar_file_path):
#                 download_url(URL, tar_file_path)
#             print('unzip downloaded dataset: %s' % tar_file_path)
#             tar = tarfile.open(tar_file_path, 'r:xz')
#             tar.extractall(self.mvtec_folder_path)
#             tar.close()

#         return


# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)


# def download_url(url, output_path):
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
