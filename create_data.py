import os,sys
os.chdir(sys.path[0])

import joblib
from tqdm import tqdm

from creat_utils.loader import *
from sklearn.preprocessing import StandardScaler


# 生成数据列表
def get_data_list(audio_path, list_path):
    #子目录
    aps = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
           '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']

    #创建统计文件
    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')
    label_first = True

    for path in aps:
        audio_p0 = os.path.join(audio_path, path)
        audios = os.listdir(audio_p0)
        #angry-0,happy-1,neutral-2,sad-3,surprise-4
        label_idx = 0
        for i in range(len(audios)):
            if(audios[i][-1]=='t'):
                continue
            
            if(label_first):
                f_label.write(f'{audios[i]}\n')
            
            p_sounds = os.path.join(audio_p0, audios[i])

            train_sounds = os.listdir(os.path.join(p_sounds, "train"))
            for sound in train_sounds:
                sound_path = os.path.join(p_sounds, "train", sound).replace('\\', '/')
                f_train.write('%s\t%d\n' % (sound_path, label_idx))

            test_sounds = os.listdir(os.path.join(p_sounds, "test"))
            for sound in test_sounds:
                sound_path = os.path.join(p_sounds, "test", sound).replace('\\', '/')
                f_test.write('%s\t%d\n' % (sound_path, label_idx))
            
            label_idx += 1
        
        if(label_first):
            label_first = False
            f_label.close()

    f_train.close()
    f_test.close()


# 生成归一化文件
def create_standard(list_path, scaler_path):
    #遍历所有用于训练的音频
    with open(os.path.join(list_path, 'train_list.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    #获取其音频特征
    data = []
    for line in tqdm(lines):
        path, label = line.split('\t')
        data.append(load_audio(path, mode='infer'))
    #拟合数据
    scaler = StandardScaler().fit(data)
    #存储在指定路径下
    joblib.dump(scaler, scaler_path)


if __name__ == '__main__':
    #读取ESD
    get_data_list('ESD', 'ESD')
    create_standard('ESD', 'ESD/standard.m')
