import os,sys
os.chdir(sys.path[0])

import argparse
import functools

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from data_utils.myDataset import *
from modules.model import Model
from data_utils.args_tip import *
from eval_utils.matrix import *


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                        '训练的批量大小')
add_arg('num_workers',      int,    4,                         '读取数据的线程数量')
add_arg('num_class',        int,    5,                         '分类的类别数量')
add_arg('test_list_path',   str,    'ESD/test_list.txt',       '测试数据的数据列表路径')
add_arg('label_list_path',   str,   'ESD/label_list.txt',      '标签列表路径')
add_arg('scaler_path',      str,    'ESD/standard.m',          '测试数据的数据列表路径')
add_arg('model_path',       str,    'output/models/model.pth', '模型保存的路径')
add_arg('device_type',      str,    'cpu',                     '设备类型 cuda/cpu')
args = parser.parse_args()


def evaluate():
    # 获取评估数据
    eval_dataset = CustomDataset(args.test_list_path,
                                 scaler_path=args.scaler_path,
                                 mode='eval',
                                 sr=16000,
                                 chunk_duration=3)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    # 获取分类标签
    with open(args.label_list_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        class_labels = []
        for line in lines:
            class_labels.append(line.replace('\n', ''))

    # 获取模型
    device = torch.device(args.device_type)
    model = Model(num_class=args.num_class)
    # 读取训练好的模型参数并加载到模型中
    weights = torch.load(args.model_path, map_location=torch.device(args.device_type))
    model.load_state_dict(weights)
    model.to(device)

    # 开始进行评估
    model.eval()
    print('开始评估...')
    pred_labels, ture_labels = [], []
    for batch, (audio_feature, ture_label) in enumerate(eval_loader):
        audio_feature = audio_feature.to(device)
        # 进行预测，得到每种标签的概率分布
        output = model(audio_feature)
        # 取概率最大的标签为最终预测标签
        pred_label = np.argmax(output.data.cpu().numpy(), axis=1)
        pred_labels.extend(pred_label)
        # 真实标签
        ture_labels.extend(ture_label.data.cpu().numpy().tolist())

    # 获取分类报告
    print(classification_report(ture_labels, pred_labels, target_names=class_labels, digits=4))
    # 获取并绘制混淆矩阵
    matrix = confusion_matrix(ture_labels, pred_labels)
    # print(matrix)
    plot_confusion_matrix(matrix=matrix, class_labels=class_labels, type='recall')


if __name__ == '__main__':
    print_arguments(args)
    evaluate()