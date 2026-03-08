import os

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(matrix, class_labels, type='precision', save_path = 'output/confusion_matrix.png', show=False):
    total = np.sum(matrix)      # 测试样本的总数
    TP_sum = 0
    num_class = len(class_labels)
    # 统计混淆矩阵对角线的元素数之和，即分类正确的数量
    for i in range(num_class):
        TP_sum += matrix[i, i]
    acc = TP_sum / total        # 总体准确率

    # prob与混淆矩阵同样结构,记录目标被分为各类别的概率
    rows, cols = np.shape(matrix)
    prob = np.zeros((rows, cols))
    for x in range(num_class):
        for y in range(num_class):
            # 计算精确度或召回率
            if type == 'recall':
                prob_xy = matrix[y][x] / (np.sum(matrix[y, :]) + 1e-6)
            else:
                prob_xy = matrix[y][x] / (np.sum(matrix[:, x]) + 1e-6)
            prob[y, x] = prob_xy

    # 设置标题
    plt.title('Confusion matrix(' + type + ')\naccuracy={:.4f}'.format(acc))
    # 设置x、y轴坐标和对应label
    plt.xticks(range(num_class), class_labels, rotation=45)
    plt.xlabel('Predicted Labels')
    plt.yticks(range(num_class), class_labels)
    plt.ylabel('True Labels')
    # 概率越大格子颜色越深
    plt.imshow(prob, cmap=plt.cm.Blues)
    plt.colorbar()

    for x in range(num_class):
        for y in range(num_class):
            # 标注概率值，概率过小的视为0
            prob_xy = prob[y, x]
            if prob_xy < 1e-2:
                plt.text(x, y, '0', va='center', ha='center', color='black')
            else:
                plt.text(x, y, "%0.3f" % (prob_xy,), va='center', ha='center',
                     color="white" if prob_xy > 0.15 else "black")

    plt.tight_layout()
    # 保存生成的图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    print("对应混淆矩阵已生成在" + save_path)
    if show:
        plt.show()        # 显示图片
