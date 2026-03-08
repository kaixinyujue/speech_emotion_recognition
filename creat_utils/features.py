import librosa
import numpy as np

#获取音频特征
def audio_features(X, sample_rate: float) -> np.ndarray:
    stft = np.abs(librosa.stft(X))  #得到频谱图

    # fmin 和 fmax 对应于人类语音的最小最大基本频率
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])
    
    #计算音高的调整偏移、均值、标准差、最大值
    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)
    pitch_max = np.max(pitch)

    # 频谱质心
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)  #归一化处理
    #计算均值、标准差和最大值
    cent_mean = np.mean(cent)
    cent_std = np.std(cent)
    cent_max = np.max(cent)

    # 谱平面
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # 使用系数为50的MFCC特征
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccs_std = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfcc_max = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # 色谱图
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # 梅尔频率
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    # ottava对比
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # 过零率
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    # S, phase = librosa.magphase(stft)
    S = np.abs(stft)
    Magnitude_mean = np.mean(S)
    Magnitude_std = np.std(S)
    Magnitude_max = np.max(S)

    # 均方根能量
    rmse = librosa.feature.rms(S=S)[0]
    rms_mean = np.mean(rmse)
    rms_std = np.std(rmse)
    rms_max = np.max(rmse)

    features = np.array([
        flatness, zerocr, Magnitude_mean, Magnitude_max, cent_mean, cent_std,
        cent_max, Magnitude_std, pitch_mean, pitch_max, pitch_std,
        pitch_tuning_offset, rms_mean, rms_max, rms_std
    ])

    features = np.concatenate((features, mfccs, mfccs_std, mfcc_max, chroma, mel, contrast)).astype(np.float32)
    return features
