import random
import librosa
import warnings
from creat_utils.features import *
warnings.filterwarnings("ignore")


#加载并预处理音频
def load_audio(audio_path, mode='train', sr=16000, chunk_duration=3, augmentors=None):
    # 读取音频数据
    wav, sr_wav = librosa.load(audio_path, sr=sr)
    # 随机裁剪
    num_wav_samples = wav.shape[0]
    # 计算样本数
    num_chunk_samples = int(chunk_duration * sr)

    if mode == 'train':
        if num_wav_samples > num_chunk_samples + 1:
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
            wav = wav[start:stop]

            # 一定概率再次裁剪
            if random.random() > 0.5:
                crop_length = random.randint(1, sr // 4) #定义一个随机数
                wav[:crop_length] = 0 #用0填充
                wav = wav[:-crop_length] #删除部分样本

        # 数据增强
        if augmentors is not None:
            for key, augmentor in augmentors.items():
                if key == 'specaug':
                    continue
                wav = augmentor(wav)

    elif mode == 'eval':
        # 为避免显存溢出，只裁剪指定长度
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]

    # 获取音频特征
    features = audio_features(X=wav, sample_rate=sr)
    return features
