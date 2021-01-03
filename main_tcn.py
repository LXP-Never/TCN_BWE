# Author:凌逆战
# -*- encoding:utf-8 -*-
# 修改时间：2020年7月19日
import time
import os
import tensorflow as tf
import numpy as np
from models.TCN import TCN_model, dilation_conv_model
from ops import load_h5, load_model, get_num_params, tf_RMSE_MAE_loss, tf_compute_snr, tf_compute_log_distortion
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示error信息

tf.flags.DEFINE_integer('batch_size', 32, 'batch size, default: 64')
tf.flags.DEFINE_integer('epochs', 100, 'epoch，default: 10')
tf.flags.DEFINE_float('learning_rate', 3e-4, '初始学习率, 默认: 3e-4')
tf.flags.DEFINE_string('train_data', "./data_preparation/resample_data/speaker225_resample_train.h5", '保存检查点的地址')
tf.flags.DEFINE_string('val_data', "./data_preparation/resample_data/speaker225_resample_val.h5", '保存检查点的地址')
tf.flags.DEFINE_string('checkpoints_dir', "./checkpoints/TCN", '保存检查点的地址')
tf.flags.DEFINE_string('event_dir', "./event_file/TCN", 'tensorboard事件文件的地址')
FLAGS = tf.flags.FLAGS


def train():
    X_train, Y_train = load_h5(FLAGS.train_data)
    X_val, Y_val = load_h5(FLAGS.val_data)
    print("训练数据shape", X_train.shape)  # (3392, 8192, 1)
    print("测试数据shape", X_val.shape)  # (64, 8192, 1)

    batch_size = FLAGS.batch_size  # 一个batch训练多少个样本
    sample_nums = X_train.shape[0]  # 3392
    batch_nums = sample_nums // batch_size  # 一个epoch中应该包含多少batch数据
    epochs = FLAGS.epochs  # 训练周期数
    learning_rate = FLAGS.learning_rate  # 初始学习率

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
        logits = TCN_model(inputs=inputs,dropout_rate=dropout_rate, is_training=is_training)
        ############    损失函数   ############
        time_loss, frequency_loss, tf_losses_op = tf_RMSE_MAE_loss(labels=labels, logits=logits, alpha=0.85)  # 时频损失
        tf.add_to_collection('losses', tf_losses_op)
        tf_losses_op = tf.add_n(tf.get_collection("losses"))  # total_loss=模型损失+权重正则化损失
        ############    模型精度   ############
        snr_op = tf_compute_snr(labels, logits)
        lsd_op = tf_compute_log_distortion(labels, logits)
        ############    优化器   ############
        variable_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  # 可训练变量列表
        # 创建优化器，更新网络参数，最小化loss，
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,  # 初始学习率
                                                   global_step=global_step,
                                                   decay_steps=batch_nums*20,  # 多少步衰减一次 300-40
                                                   decay_rate=0.1,  # 衰减率 0.9
                                                   staircase=True)  # 以阶梯的形式衰减
        total_parameters = get_num_params()
        print("权重数量", len(variable_to_train))  # 140
        print("可训练参数的总数: %d" % total_parameters)  # 428441
        print("更新操作", len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))  # 68
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # adam优化器,adam算法好像会自动衰减学习率，
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss=tf_losses_op,
                                                                      global_step=global_step,
                                                                      var_list=variable_to_train)
        ############    TensorBoard可视化 summary  ############
        summary_writer = tf.summary.FileWriter(logdir=FLAGS.event_dir, graph=graph)  # 创建事件文件
        # tf.summary.scalar(name="tf_losses", tensor=tf_losses_op)  # 收集损失值变量
        # tf.summary.scalar(name="train_SNR", tensor=snr_op)  # 收集精度值变量
        # tf.summary.scalar(name="train_LSD", tensor=lsd_op)  # 收集精度值变量
        tf.summary.scalar(name='learning_rate', tensor=learning_rate)
        merged_summary_op = tf.summary.merge_all()  # 将所有的summary合并为一个op
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

        for epoch in range(epochs):
            train_sample_nums_index = np.arange(sample_nums)
            np.random.shuffle(train_sample_nums_index)
            # print("训练数据被打乱的索引",train_sample_nums_index)
            X_train = X_train[train_sample_nums_index]
            Y_train = Y_train[train_sample_nums_index]
            # print("打乱顺序后的训练数据shape", X_train.shape)

            for i in range(batch_nums):
                start_time = time.time()
                train_batch_X = X_train[i * batch_size:(i + 1) * batch_size]
                train_batch_Y = Y_train[i * batch_size:(i + 1) * batch_size]

                # 使用真实数据填充placeholder，运行训练模型和合并变量操作
                _, summary_op, train_loss, train_snr, train_lsd, step = sess.run(
                    [train_op, merged_summary_op, tf_losses_op, snr_op, lsd_op, global_step],
                    feed_dict={inputs: train_batch_X,
                               labels: train_batch_Y,
                               dropout_rate: 0.0,
                               is_training: True})
                if step % 53 == 0:  # 太密集tensorboard曲线波动太大了，建议改成100
                    ############    可视化打印   ############
                    print("Epoch：{}/{} batch：{}/{} train_loss：{} train_snr：{} train_lsd：{} time：{}".format(
                        epoch+1, epochs, i+1, batch_nums, train_loss, train_snr, train_lsd, time.time() - start_time))

                    # 打印一些可视化的数据，损失...
                    for j in range(X_val.shape[0] // batch_size):
                        val_batch_X = X_val[j * batch_size:(j + 1) * batch_size]
                        val_batch_Y = Y_val[j * batch_size:(j + 1) * batch_size]
                        val_loss, val_snr, val_lsd = sess.run([tf_losses_op, snr_op, lsd_op],
                                                              feed_dict={inputs: val_batch_X,
                                                                         labels: val_batch_Y,
                                                                         dropout_rate: 0.0,
                                                                         is_training: False})

                    print("\t\t\t\t\t\t val_loss：{} val_snr：{} val_lsd：{}".format(val_loss, val_snr, val_lsd))
                    objectives_summary = tf.Summary()
                    objectives_summary.value.add(tag='train_loss', simple_value=train_loss)
                    objectives_summary.value.add(tag='train_snr', simple_value=train_snr)
                    objectives_summary.value.add(tag='train_lsd', simple_value=train_lsd)
                    objectives_summary.value.add(tag='val_loss', simple_value=val_loss)
                    objectives_summary.value.add(tag='val_snr', simple_value=val_snr)
                    objectives_summary.value.add(tag='val_lsd', simple_value=val_lsd)
                    summary_writer.add_summary(objectives_summary, step)
                    summary_writer.add_summary(summary_op, step)
                    summary_writer.flush()

            ############    保存模型   ############
            if epoch % 5==0:
                save_path = saver.save(sess, save_path=os.path.join(checkpoints_dir, "model.ckpt"), global_step=step)
                tf.logging.info("模型保存在: %s" % save_path)

        print("优化完成!")


def main(argv=None):
    train()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(argv=None)
