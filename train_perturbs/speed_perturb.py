import random
import numpy as np


class SpeedPerturbAugmentor(object):
    """音频随机语速增强工具类

    :param min_speed_rate: 新采样速率下限（不应小于0.9）
    :type min_speed_rate: float
    :param max_speed_rate: 新采样速率上限（不应大于1.1）
    :type max_speed_rate: float
    :param num_rates: 速率数量
    :type num_rates: int
    :param prob: 数据增强的概率
    :type prob: float
    """

    def __init__(self, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=3, prob=0.5):
        if min_speed_rate < 0.9:
            raise ValueError("Sampling speed below 0.9 can cause unnatural effects")
        if max_speed_rate > 1.1:
            raise ValueError("Sampling speed above 1.1 can cause unnatural effects")
        self.prob = prob
        self.min_speed_rate = min_speed_rate
        self.max_speed_rate = max_speed_rate
        self.num_rates = num_rates
        if num_rates > 0:
            self.speed_rates = np.linspace(min_speed_rate, max_speed_rate, num=num_rates, endpoint=True)

    def __call__(self, waveform):
        """改变音频语速

        :param waveform: Librosa 读取的音频数据
        :type waveform: ndarray
        :return: 增强后的音频数据
        :rtype: ndarray
        """
        if random.random() > self.prob:
            return waveform
        if self.num_rates < 0:
            speed_rate = random.uniform(self.min_speed_rate, self.max_speed_rate)
        else:
            speed_rate = random.choice(self.speed_rates)
        if speed_rate == 1.0:
            return waveform

        old_length = waveform.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        waveform = np.interp(new_indices, old_indices, waveform)
        return waveform
