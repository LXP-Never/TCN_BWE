from scipy import interpolate
from scipy.signal import decimate
import librosa
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示符号

y_wb, _ = librosa.load("./resample_train_wb.wav", sr=16000, mono=True)
y_pre, _ = librosa.load("./resample_train_nb.wav", sr=16000, mono=True)


plt.subplot(2, 1, 1)
plt.specgram(y_wb, Fs=16000, scale_by_freq=True, sides='default')
plt.xlabel('宽带音频')

plt.subplot(2, 1, 2)
plt.specgram(y_pre, Fs=16000, scale_by_freq=True, sides='default')
plt.xlabel('窄带音频')



plt.show()


