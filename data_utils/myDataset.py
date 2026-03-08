import sys

import warnings
from datetime import datetime
import joblib
from creat_utils.loader import *
warnings.filterwarnings("ignore")

import numpy as np
from torch.utils import data


# 数据加载器
class CustomDataset(data.Dataset):
    def __init__(self, data_list_path, scaler_path, mode='train', sr=16000, chunk_duration=3, augmentors=None):
        super(CustomDataset, self).__init__()
        # 当预测时不需要获取数据
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.augmentors = augmentors
        self.scaler = joblib.load(scaler_path)

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].replace('\n', '').split('\t')
            # 加载并预处理音频
            features = load_audio(audio_path, mode=self.mode, sr=self.sr,
                                  chunk_duration=self.chunk_duration, augmentors=self.augmentors)
            # 转化为二维数组后进行归一化
            features = self.scaler.transform(features.reshape(1, -1))
            features = np.squeeze(features).astype(np.float32)
            label = np.array(int(label), dtype=np.int64)
            return features, label
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)
