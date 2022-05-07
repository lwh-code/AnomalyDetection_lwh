import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter

from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('Lwh_code')
    #对应的几个路径
    parser.add_argument('--data_path', type=str, default='D:\\datasets\\mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


# 从这里开始
def main():

    args = parse_args()

    #使用不同的模型的时候需要进行的处理
    """
    t_d 代表的是 总维度
    d 代表的是 将要降到的维度
    """
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device) # 将其搬到GPU上去加速
    model.eval()   #将模型进行测试
    random.seed(1024)
    torch.manual_seed(1024)  #生成一个手动生成的随机数
    if use_cuda:         #GPU可用时将该任务交给GPU
        torch.cuda.manual_seed_all(1024)

    #在此0-t_d中选取d个随机数，这里就是对维度进行了减少
    idx = torch.tensor(sample(range(0, t_d), d))

    # 模型的中间结果
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    #将每一层注册前向计算的hook，也就是综合几个层的参数
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    # exist_ok的作用是判断如果当前文件夹存在就不创建
    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]  #这里时进行AUROC图像的绘制

    total_roc_auc = []        #对整体情况进行统计，用于计算
    total_pixel_roc_auc = []

    #对规定的几种大类进行遍历
    for class_name in mvtec.CLASS_NAMES:

        #将数据加载到项目中来
        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=False)# 内存充足时使用锁页内存，如果不充足，不推荐使用
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=False)

        #这两个参数的作用，将字典进行了排序然后保存，将每一层的参数进行
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # 抽取训练集的特征，将特征进行一定的保存，保存到具体的类
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)

        #如果没有存在以下路径
        if not os.path.exists(train_feature_filepath):
            #调用进度条库，对抽取图像过程进行实时输出
            #tqdm返回的参数是=>
            #传入的参数是=> 按照训练集的大小进行进度条的跳跃，将批次改变，则进度条上会看到效果
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # 这里得到模型预测值
                #no_grad方法是对接下来的进行原地操作，也就是不会造成梯度的更新
                with torch.no_grad():
                    _ = model(x.to(device))
                # 得到中间层的输出，zip函数的用法，返回元组列表，将两个参数的对应元素打包起来
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # 初始化hook函数的输出
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # 这里将特征向量进行了concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # 随机选择d个维度
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # 这里开始进行多元高斯分布参数的学习
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)   #进行展平操作
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)  #这是创建一个方阵，以通道数作为方阵的维度
            for i in range(H * W):    #这里在进行高斯分布协方差的学习
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # 将学习到的均值和方差进行保存
            train_outputs = [mean, cov]
            #将训练集中得到的正常分布保存到文件中
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:  #如果这个文件夹存在，则代表特征已经成功被提取
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # 抽取测试集的特征
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # 将向量进行连接
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # 随机选择维度
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # 计算马氏距离矩阵
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        # 线性计算分数然后再还原回去
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        #巧妙的用采样的方法将score_map描绘了出来
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # 使用高斯方法对score map进行平滑操作
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    # 将图片进行保存
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


#对预测的mask进行绘制
def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()  #真实缺陷位置图
        heat_map = scores[i] * 255    # 热度图
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0   # 这是模型预测的mask进行绘制
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255  #变成全黑全白
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')

        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)

        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        #将图片进行保存
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()

#进行逆正则化
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


# 对各个层次的语义部分进行整合
def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

if __name__ == '__main__':
    main()
