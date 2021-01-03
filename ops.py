# Author:凌逆战
# -*- coding:utf-8 -*-
import os
import re

import h5py
import librosa
import numpy as np
import tensorflow as tf


def load_h5(h5_path):
    # load training data
    with h5py.File(h5_path, 'r') as hf:
        print('List of arrays in input file:', hf.keys())
        X = np.array(hf.get('data'), dtype=np.float32)
        Y = np.array(hf.get('label'), dtype=np.float32)
    return X, Y


def load_model(sess, saver, checkpoint_dir):
    """加载模型，
    如果模型存在，返回True和模型的step
    如果模型不存在，返回False并设置step=0"""

    # 通过checkpoint找到模型文件名
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # 返回最新的chechpoint文件名 model.ckpt-1000
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        # counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))  # 1000
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] 成功恢复模型 {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] 找不到checkpoint")
        return False, 0


def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    return total_parameters


def compute_lsp_loss(labels, logits):
    """ 均方误差"""
    mse_loss = tf.reduce_mean(labels - logits)
    # loss = tf.reduce_mean(tf.abs(labels - logits))  # tf_loss论文
    return mse_loss


def time_MSE_loss(labels, logits):
    """ 均方误差
    labels:batch_labels
    logits:batch_logits
    """
    loss = tf.reduce_mean((labels - logits) ** 2)
    return loss


def time_RMSE_loss(labels, logits):
    """ 均方根误差，
    labels:batch_labels  (batch_size, 8192,1)
    logits:batch_logits
    """
    sqrt_l2_loss = tf.sqrt(tf.reduce_mean((logits - labels) ** 2 + 1e-6, axis=[1, 2]))
    loss = tf.reduce_mean(sqrt_l2_loss, axis=0)
    return loss


def time_MAE_loss(labels, logits):
    """ 平均绝对值误差
    labels:batch_labels
    logits:batch_logits
    """
    loss = tf.reduce_mean(tf.abs(labels - logits))
    return loss


def stft_log_spectrogram(input, frame_length=256, frame_step=128, fft_length=256, window_fn=tf.signal.hann_window):
    # input:[..., samples]
    specgram = tf.signal.stft(input, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length,
                              window_fn=window_fn)
    # [..., frames, fft_unique_bins]
    magnitude_spectrograms = tf.abs(specgram)
    # log_spectrograms = tf.log(magnitude_spectrograms + 1e-6)  # 对数谱
    # [..., frames, fft_unique_bins]
    return magnitude_spectrograms


def MSE_frequency_loss(labels, logits):
    """ 均方根误差，频域损失
    labels:batch_labels
    logits:batch_logits
    """
    labels = tf.squeeze(labels, axis=2)  # 去掉位置索引为2维数为1的维度 （batch_size, input_size）
    logits = tf.squeeze(logits, axis=2)  # 去掉位置索引为2维数为1的维度

    # [batch_size, frames, fft_unique_bins]
    frequency_labels = stft_log_spectrogram(labels, frame_length=256, frame_step=128,
                                            fft_length=256, window_fn=tf.signal.hann_window)
    frequency_logits = stft_log_spectrogram(logits, frame_length=256, frame_step=128,
                                            fft_length=256, window_fn=tf.signal.hann_window)

    loss = tf.reduce_mean((frequency_labels - frequency_logits) ** 2)
    return loss


def RMSE_frequency_loss(labels, logits):
    """ 均方根误差，频域损失
    labels:batch_labels
    logits:batch_logits
    """
    labels = tf.squeeze(labels, axis=2)  # 去掉位置索引为2维数为1的维度 （batch_size, input_size）
    logits = tf.squeeze(logits, axis=2)  # 去掉位置索引为2维数为1的维度

    # [batch_size, frames, fft_unique_bins]
    frequency_labels = stft_log_spectrogram(labels, frame_length=256, frame_step=128,
                                                 fft_length=256, window_fn=tf.signal.hann_window)
    frequency_logits = stft_log_spectrogram(logits, frame_length=256, frame_step=128,
                                                 fft_length=256, window_fn=tf.signal.hann_window)

    loss = tf.sqrt(tf.reduce_mean((frequency_logits - frequency_labels) ** 2 + 1e-6, axis=[1, 2]))
    loss = tf.reduce_mean(loss, axis=0)
    return loss


def MAE_frequency_loss(labels, logits):
    """ 平均绝对值误差，频域损失
    labels:batch_labels
    logits:batch_logits
    """
    labels = tf.squeeze(labels, axis=2)  # 去掉位置索引为2维数为1的维度 （batch_size, input_size）
    logits = tf.squeeze(logits, axis=2)  # 去掉位置索引为2维数为1的维度

    # [batch_size, frames, fft_unique_bins]
    frequency_labels = stft_log_spectrogram(labels, frame_length=256, frame_step=128,
                                                 fft_length=256, window_fn=tf.signal.hann_window)
    frequency_logits = stft_log_spectrogram(logits, frame_length=256, frame_step=128,
                                                 fft_length=256, window_fn=tf.signal.hann_window)
    loss = tf.reduce_mean(tf.abs(frequency_labels - frequency_logits))
    return loss


def tf_RMSE_MAE_loss(labels, logits, alpha=0.85):
    """
    labels:batch_labels [batch_size, input_size, 1]
    logits:batch_logits  [batch_size, input_size, 1]
    """
    # ##### 时域损失 RMSE(平均绝对值误差)
    RMSE_time_loss = time_RMSE_loss(labels, logits)  # Kuleshov论文

    # ##### 频域损失
    labels = tf.squeeze(labels, axis=2)  # 去掉位置索引为2维数为1的维度 （batch_size, input_size）
    logits = tf.squeeze(logits, axis=2)  # 去掉位置索引为2维数为1的维度

    # [batch_size, frames, fft_unique_bins]
    frequency_labels = stft_log_spectrogram(labels, frame_length=256, frame_step=128, fft_length=256,
                                            window_fn=tf.signal.hann_window)
    frequency_logits = stft_log_spectrogram(logits, frame_length=256, frame_step=128, fft_length=256,
                                            window_fn=tf.signal.hann_window)
    MAE_frequency_loss = tf.reduce_mean(tf.abs(frequency_labels - frequency_logits))
    return RMSE_time_loss, MAE_frequency_loss, alpha * RMSE_time_loss + (1 - alpha) * MAE_frequency_loss


def tf_compute_snr(labels, logits):
    signal = tf.reduce_mean(labels ** 2, axis=[1, 2])
    noise = tf.reduce_mean((logits - labels) ** 2 + 1e-6, axis=[1, 2])
    snr = 10 * tf.log(signal / noise + 1e-8) / tf.log(10.)
    avg_snr = tf.reduce_mean(snr, axis=0)
    return avg_snr


def tf_get_power(x):
    # input:[..., samples]
    x = tf.squeeze(x, axis=2)  # 去掉位置索引为2维数为1的维度 （batch_size, input_size）
    S = tf.signal.stft(x, frame_length=2048, frame_step=512, fft_length=2048,
                       window_fn=tf.signal.hann_window)
    # [..., frames, fft_unique_bins] (16, 13, 1025)
    S = tf.log(tf.abs(S) ** 2 + 1e-8)
    return S


def tf_compute_log_distortion(labels, logits):
    S1 = tf_get_power(labels)  # [..., frames, fft_unique_bins]
    S2 = tf_get_power(logits)
    original_target_squared = (S1 - S2) ** 2

    lsd = tf.reduce_mean(tf.sqrt(tf.reduce_mean(original_target_squared, axis=2)), axis=1)
    lsd = tf.reduce_mean(lsd, axis=0)
    return lsd
