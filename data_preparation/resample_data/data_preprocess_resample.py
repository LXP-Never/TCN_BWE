# Author：凌逆战
# -*- coding: utf-8 -*-
"""
方法：重采样，高频部分不会恢复，时间维度对不上，因此在重采样之前需要给原音频裁切取整
得到训练数据为8000Hz，Ground True为16kHz。
"""
import fnmatch
import os
import h5py
import librosa
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--sr', type=int, default=16000, help='音频采样率')
parser.add_argument('--wav_dir', default="F:/dataset/VCTK-Corpus/wav48/p225", help='存放wav文件的路径')
parser.add_argument('--h5_dir', default="./single_speaker225_resample_r=2.h5", help='输出 h5存档的路径')
parser.add_argument('--scale', type=int, default=2, help='缩放因子')  # 2、4、6
parser.add_argument('--dimension', type=int, default=8192, help='patch的维度')
parser.add_argument('--stride', type=int, default=4096, help='提取patch时候的步幅')
parser.add_argument('--batch_size', type=int, default=64, help='我们产生的 patches 是batch size的倍数')
args = parser.parse_args()

# 如果是TIMIT数据集
# train_set_shape:(48576, 8192, 1)
# test_set_shape:(17728, 8192, 1)
# python data_preprocess_resample.py --wav_dir "F:/dataset/TIMIT/TRAIN" --h5_dir "./TIMIT_resample_train_r=2.h5"
# python data_preprocess_resample.py --wav_dir "F:/dataset/TIMIT/TEST" --h5_dir "./TIMIT_resample_test_r=2.h5"

def preprocess(args, h5_file, save_wav):
    # 列出所有要处理的文件 列表
    wav_list = []
    for root, dirnames, filenames in os.walk(args.wav_dir):
        for filename in fnmatch.filter(filenames, "*.wav"):  # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
            wav_list.append(os.path.join(root, filename))
    num_files = len(wav_list)  # num_files音频文件的个数
    print("音频的个数为：", num_files)

    # patches to extract and their size / 要提取的补丁及其大小
    dim = args.dimension  # patch的维度 default=8192
    wb_stride = args.stride  # 提取patch时候的步幅 default=3200

    wb_patches = list()  # 宽带音频补丁空列表
    nb_patches = list()  # 窄带音频补丁空列表

    for j, wav_path in enumerate(wav_list):
        if j % 10 == 0:  # 每隔10次打印一下文件的索引和文件路径名
            print('%d/%d' % (j, num_files))

        wb_wav, _ = librosa.load(wav_path, sr=args.sr)  # 加载音频文件 采样率 sr = 16000

        # 裁剪，使其与缩放比率一起工作，结果:能被缩放比例整除，因为不能整除的已经被减去了
        wav_len = len(wb_wav)
        wb_wav = wb_wav[: wav_len - (wav_len % args.scale)]

        # 生成低分辨率版本
        nb_wav = librosa.core.resample(wb_wav, args.sr, args.sr / args.scale)  # 下采样率 16000-->8000
        nb_wav = librosa.core.resample(nb_wav, args.sr / args.scale, args.sr)  # 上采样率 8000-->16000，并不恢复高频部分

        # 生成补丁
        max_i = len(wb_wav) - dim + 1
        for i in range(0, max_i, wb_stride):
            wb_patch = np.array(wb_wav[i: i + dim])
            nb_patch = np.array(nb_wav[i: i + dim])

            wb_patches.append(wb_patch.reshape((dim, 1)))
            nb_patches.append(nb_patch.reshape((dim, 1)))

    # 裁剪补丁，使其成为小批量的倍数
    num_wb_patches = len(wb_patches)
    num_nb_patches = len(nb_patches)
    print("num_wb_patches", num_wb_patches)  # 852
    print("num_nb_patches", num_nb_patches)  # 852

    print('batch_size:', args.batch_size)  # batch_size: 64
    # num_wb_patches要能够被batch整除，保留能够被整除的，这样才能保证每个样本都能被训练到
    num_to_keep_wb = num_wb_patches // args.batch_size * args.batch_size
    wb_patches = np.array(wb_patches[:num_to_keep_wb])

    num_to_keep_nb = num_nb_patches // args.batch_size * args.batch_size
    nb_patches = np.array(nb_patches[:num_to_keep_nb])

    print('hr_patches shape:', wb_patches.shape)  # (832, 16384, 1)
    print('lr_patches shape:', nb_patches.shape)  # (832, 16384, 1)

    # 创建 hdf5 文件
    data_set = h5_file.create_dataset('data', nb_patches.shape, np.float32)  # lr
    label_set = h5_file.create_dataset('label', wb_patches.shape, np.float32)  # hr

    data_set[...] = nb_patches  # ...代替了前面两个冒号, data_set[...]=data_set[:,:]
    label_set[...] = wb_patches

    if save_wav:
        librosa.output.write_wav('resample_train_wb.wav', wb_patches[40].flatten(), args.sr, norm=False)
        librosa.output.write_wav('resample_train_nb.wav', nb_patches[40].flatten(), args.sr, norm=False)
        print(wb_patches[40].shape)  # (8192, 1)
        print(nb_patches[40].shape)  # (8192, 1)
        print('保存了两个示例')


if __name__ == '__main__':
    # 创造训练
    with h5py.File(args.h5_dir, 'w') as f:
        preprocess(args, f, save_wav=True)
