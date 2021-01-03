# Author:凌逆战
# -*- coding:utf-8 -*-
from pystoi import stoi
import fnmatch
import os
import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
# #### 超参数

label_wav_dir = "./label"
logits_wav_dir = "./logits"
sr = 16000


# #####################
def get_power(x):
    S = librosa.stft(x, 2048)   # (1 + n_fft/2, n_frames)
    # p = np.angle(S)
    S = np.log10(np.abs(S) ** 2)
    return S


def compute_log_distortion(labels, logits):
    S1 = get_power(labels)
    S2 = get_power(logits)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2, axis=0)), axis=0)
    return lsd


def compute_snr(labels, logits):
    sqrt_l2_loss = np.mean((logits - labels) ** 2)
    sqrn_l2_norm = np.mean(labels ** 2)
    snr = 10 * np.log10(sqrn_l2_norm / sqrt_l2_loss)
    return snr


def load_wav(wav_dir):
    wav_list = []
    for root, dirnames, filenames in os.walk(wav_dir):
        for filename in fnmatch.filter(filenames, "*.wav"):  # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
            wav_list.append(os.path.join(root, filename))
    return wav_list


label_wav_list = load_wav(label_wav_dir)
logits_wav_list = load_wav(logits_wav_dir)

print(label_wav_list)  # ['./DNN/r=4/label\\p225_355.wav', './DNN/r=4/label\\p225_356.wav',...
print(logits_wav_list)  # ['./DNN/r=4\\0.wav', './DNN/r=4\\1.wav',...

lsd_list, snr_list, stoi_list = list(), list(), list()
for label_wav, logits_wav in zip(label_wav_list, logits_wav_list):
    print("正在处理的音频")

    label, _ = librosa.load(label_wav, sr=sr)
    logits, _ = librosa.load(logits_wav, sr=sr)

    len_label = len(label)
    len_logits = len(logits)
    # print(len_label)
    # print(len_logits)
    # assert len_label ==len_logits, "长度不相等"
    min_len = min(len_label,len_logits)
    label = label[:min_len]
    logits = logits[:min_len]

    # ###################
    fig = plt.figure()
    gca = plt.gca()
    # gca.set_position([0.1, 0.1, 0.9, 0.9])
    norm = matplotlib.colors.Normalize(vmin=-200, vmax=-40)
    plt.subplot(2, 1, 1)
    plt.specgram(label, Fs=16000, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
    plt.subplots_adjust(left=0.126, right=0.870, top=0.938, bottom=0.115)
    plt.subplots_adjust(wspace=0.2, hspace=0.689)  # 调整子图间距
    plt.title("宽带音频",fontsize=15)
    plt.xlabel('秒/s',fontsize=15)
    plt.ylabel('频率/Hz',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.subplot(2, 1, 2)
    plt.specgram(logits, Fs=16000, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
    plt.subplots_adjust(left=0.126, right=0.870, top=0.938, bottom=0.115)
    plt.subplots_adjust(wspace=0.2, hspace=0.689)  # 调整子图间距
    plt.title("重构宽带音频",fontsize=15)
    plt.xlabel('秒/s',fontsize=15)
    plt.ylabel('频率/Hz',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    l = 0.88    # 左边
    b = 0.115    # 底部
    w = 0.015   # 右
    h = 0.82   # 高
    # 对应 l,b,w,h；设置colorbar位置；
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    cbar = plt.colorbar(norm=norm, cax=cbar_ax, format="%+2.f dB")  # -200 -50
    # cbar.set_label('能量')
    # plt.tight_layout()
    plt.show()

    lsd = compute_log_distortion(label, logits)
    snr = compute_snr(label, logits)
    stoi_score = stoi(label, logits, 16000, extended=False)
    lsd_list.append(lsd)
    snr_list.append(snr)
    stoi_list.append(stoi_score)
    print("对数谱距离", lsd)
    print("信噪比", snr)
    print("可懂度", stoi_score)

lsd_avg = np.mean(lsd_list)
snr_avg = np.mean(snr_list)
stoi_avg = np.mean(stoi_list)
print("最终的对数谱距离", lsd_avg)  # 3.6951742
print("最终的信噪比", snr_avg)  # 20.804820203469493
print("最终的可懂度", stoi_avg)