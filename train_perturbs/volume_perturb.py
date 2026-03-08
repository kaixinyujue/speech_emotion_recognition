import random


class VolumePerturbAugmentor(object):
    """添加随机音量大小

    :param min_gain_dBFS: 最小增益
    :type min_gain_dBFS: int
    :param max_gain_dBFS: 最小增益大
    :type max_gain_dBFS: int
    :param prob: 数据增强的概率
    :type prob: float
    """

    def __init__(self, min_gain_dBFS=-15, max_gain_dBFS=15, prob=0.5):
        self.prob = prob
        self.min_gain_dBFS = min_gain_dBFS
        self.max_gain_dBFS = max_gain_dBFS

    def __call__(self, waveform):
        """改变音频音量大小

        :param waveform: Librosa 读取的音频数据
        :type waveform: ndarray
        :return: 增强后的音频数据
        :rtype: ndarray
        """
        if random.random() > self.prob:
            return waveform

        gain = random.uniform(self.min_gain_dBFS, self.max_gain_dBFS)
        gain_linear = 10 ** (gain / 20.0)  # 转换为线性增益

        return waveform * gain_linear
