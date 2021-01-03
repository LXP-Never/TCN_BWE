from scipy import interpolate
from scipy.signal import decimate
import librosa
from scipy import signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示符号

y_wb, _ = librosa.load("./label/p225_355.wav", sr=16000, mono=True)
y_nb, _ = librosa.load("./nb/p225_355_NB.WAV", sr=16000, mono=True)
y_pre, _ = librosa.load("./logits/p225_355_NB.wav", sr=16000, mono=True)


# ###################
fig = plt.figure(figsize=(15,7))
gca = plt.gca()
# gca.set_position([0.1, 0.1, 0.9, 0.9])
norm = matplotlib.colors.Normalize(vmin=-200, vmax=-40)
plt.subplot(1, 3, 1)
plt.title("窄带音频",fontsize=15)
plt.specgram(y_nb, Fs=16000, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
plt.xlabel('秒/s',fontsize=15)
plt.ylabel('频率/Hz',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplots_adjust(top=0.963, bottom=0.080, left=0.056, right=0.940)
plt.subplots_adjust(hspace=0.474, wspace=0.290)  # 调整子图间距

plt.subplot(1, 3, 2)
plt.title("宽带音频",fontsize=15)
plt.specgram(y_wb, Fs=16000, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
plt.xlabel('秒/s',fontsize=15)
plt.ylabel('频率/Hz',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplots_adjust(top=0.963, bottom=0.080, left=0.056, right=0.940)
plt.subplots_adjust(hspace=0.474, wspace=0.290)  # 调整子图间距

plt.subplot(1, 3, 3)
plt.title("预测音频",fontsize=15)
plt.specgram(y_pre, Fs=16000, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
plt.subplots_adjust(left=0.101, right=0.910, top=0.943, bottom=0.115)
plt.subplots_adjust(wspace=0.2, hspace=0.474)  # 调整子图间距
plt.xlabel('秒/s',fontsize=15)
plt.ylabel('频率/Hz',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplots_adjust(top=0.963, bottom=0.080, left=0.056, right=0.940)
plt.subplots_adjust(hspace=0.474, wspace=0.290)  # 调整子图间距

l = 0.95    # 左边
b = 0.115    # 底部
w = 0.009   # 右
h = 0.82   # 高
# 对应 l,b,w,h；设置colorbar位置；
rect = [l, b, w, h]
cbar_ax = fig.add_axes(rect)
plt.colorbar(norm=norm, cax=cbar_ax, format="%+2.f dB")  # -200 -50
# plt.tight_layout()

plt.show()


