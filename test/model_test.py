# Author:凌逆战
# -*- encoding:utf-8 -*-
# 修改时间：2020年7月19日
import fnmatch
import time
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示error信息
import librosa
import tensorflow as tf
import numpy as np

from models.TCN import TCN_model, dilation_conv_model
from ops import load_h5, load_model, get_num_params, tf_RMSE_MAE_loss, \
    tf_compute_snr, tf_compute_log_distortion

tf.flags.DEFINE_integer('batch_size', 64, 'batch size, default: 64')
tf.flags.DEFINE_integer('epochs', 10, 'epoch，default: 10')
tf.flags.DEFINE_float('learning_rate', 3e-4, '初始学习率, 默认: 3e-4')
tf.flags.DEFINE_string('train_data', "../data_preparation/resample_data/speaker225_resample_train.h5", '保存检查点的地址')
tf.flags.DEFINE_string('val_data', "../data_preparation/resample_data/speaker225_resample_val.h5", '保存检查点的地址')
tf.flags.DEFINE_string('checkpoints_dir', "../checkpoints/TCN", '保存检查点的地址')
tf.flags.DEFINE_string('event_dir', "../event_file/TCN", 'tensorboard事件文件的地址')
FLAGS = tf.flags.FLAGS


def val():
    wb_path = './label'

    ############    保存检查点的地址   ############
    checkpoints_dir = FLAGS.checkpoints_dir  # ./checkpoints/rnn_model
    # 如果检查点不存在，则创建
    if not os.path.exists(checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)

    ######################################################
    #                    创建图                          #
    ######################################################
    graph = tf.Graph()  # 自定义图
    # 在自己的图中定义数据和操作
    with graph.as_default():
        inputs = tf.placeholder(dtype="float", shape=[None, None, 1], name='inputs')  # (batch_size, 8192, 1)
        labels = tf.placeholder(dtype="float", shape=[None, None, 1], name='labels')
        dropout_rate = tf.placeholder(dtype="float", name='dropout_rate')  # 0.1
        is_training = tf.placeholder(dtype="bool", name='is_training')
        # is_training = tf.placeholder(tf.bool)
        ############    搭建模型   ############
        logits = TCN_model(inputs=inputs, dropout_rate=dropout_rate, is_training=is_training)
        ############    模型保存和恢复 Saver   ############
        saver = tf.train.Saver(max_to_keep=3)

    ######################################################
    #                   创建会话                         #
    ######################################################
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config, graph=graph) as sess:
        # 加载模型，如果模型存在返回 是否加载成功和训练步数
        could_load, checkpoint_step = load_model(sess, saver, FLAGS.checkpoints_dir)
        if could_load:
            print(" [*] 模型加载成功")
        else:
            print(" [!] 模型加载失败")
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

        print("之前模型的step", checkpoint_step)

        for file in os.listdir(wb_path):  # os.listdir返回指定的文件夹包含的文件或文件夹的名字的列表
            if fnmatch.fnmatch(file, '*.wav'):  # 判断是否有后缀为.py的文件，*代表文件名长度格式不限制。
                print(os.path.join(wb_path, file))
                wb_audio, _ = librosa.load(path=os.path.join(wb_path, file), sr=16000)
                nb_audio = librosa.core.resample(wb_audio, 16000, 8000)
                nb_audio = librosa.core.resample(nb_audio, 8000, 16000)

                nb_audio = nb_audio[:len(nb_audio)-len(nb_audio)%2]
                print("音频的长度是不是偶数", len(nb_audio))
                nb_audio = nb_audio.reshape(1, -1, 1)
                prdict_wav = sess.run(logits,
                                      feed_dict={inputs: nb_audio,
                                                 dropout_rate: 0.0,
                                                 is_training: False})
                prdict_wav = prdict_wav.flatten()
                librosa.output.write_wav(path=os.path.join("./logits", file), y=prdict_wav, sr=16000, norm=False)

def main(argv=None):
    val()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(argv=None)
