# coding: utf-8
 
  
import tensorflow as tf
import keras.backend as K
import keras 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import time

def read_data(snr):
    # 读取数据
    input_data = pd.read_csv(os.path.join('data', 'input%d.csv' % snr), header=None)
    output_data = pd.read_csv(os.path.join('data', 'output%d.csv' % snr), header=None)

    x_train, x_test, y_train, y_test = train_test_split(input_data.T,
                                                        output_data.T,
                                                        test_size=test_rate)
    
    return x_train.values, x_test.values, y_train.values, y_test.values
  
INPUT_NODE = 16  # 输入节点
OUTPUT_NODE = 16  # 输出节点
BATCH_SIZE = 200  # 一批多少数据
EPOCHS = 20  # 过多少次数据
LEARNING_RATE_BASE = 0.001  # 基本学习率
STAIRCASE = True   # 学习率衰减阶梯状衰减
DROP_OUT = 0.3  # dropout率
data_num = 100000  # 数据总数
test_rate = 0.3  # 测试数据占比率
train_shape = (INPUT_NODE,int(data_num * (1-test_rate)))
test_shape = (OUTPUT_NODE,int(data_num * test_rate))
SNR = [0, 1, 2, 3, 4]
  

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配

  
sess = tf.Session(config=config)
K.set_session(sess)

# 建模
x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE))
y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE))

layer1 = keras.layers.Dense(128, activation='relu')(x)

layer1 = keras.layers.Dropout(DROP_OUT)(layer1)

layer2 = keras.layers.Dense(256, activation='relu')(layer1)

layer2 = keras.layers.Dropout(DROP_OUT)(layer2)

layer3 = keras.layers.Dense(128, activation='relu')(layer2)

layer3 = keras.layers.Dropout(DROP_OUT)(layer3)

y = keras.layers.Dense(OUTPUT_NODE)(layer3)

# model = keras.models.Sequential()
# model.add(keras.layers.normalization.BatchNormalization(input_shape=(INPUT_NODE,)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dropout(DROP_OUT))
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(DROP_OUT))
# model.add(keras.layers.Dense(128,activation='relu'))
# model.add(keras.layers.Dropout(DROP_OUT))
# model.add(keras.layers.Dense(OUTPUT_NODE, activation='softmax'))

#
# def error_num(y_ture, y_pred):
#     return keras.backend.sum(keras.backend.square(y_pred - y_ture))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
#               metrics=['accuracy', error_num])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,
                                                                 logits=y))

error_num = tf.reduce_sum(tf.square(y - y_)) / 2 / BATCH_SIZE

# 优化器
# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',  # 存储当前迭代的轮数
                              dtype=tf.int32,  # 整数
                              initializer=0,  # 初始化值
                              trainable=False)  # 不可训练

# 指数衰减学习速率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           data_num/BATCH_SIZE,
                                           0.99,
                                           staircase=STAIRCASE)

# 定义优化函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

# model.fit(x_train.values,
#           y_train.values,
#           batch_size=BATCH_SIZE,
#           epochs = EPOCHS,
#           verbose = 1,
#           validation_data = (x_test.values, y_test.values),
#           )


snr_ber = []
train_loss_list = []
test_loss_list = []
time_list = []

# 保存开始时间
time_start = time.time()

for snr in SNR:
    train_loss_snr = []
    test_loss_snr = []
    
    X_train, X_test, Y_train, Y_test = read_data(snr)

    sess.run(tf.global_variables_initializer())  # 初始化
    for epoch in range(EPOCHS):
        for i in range(data_num // BATCH_SIZE):
            # 设置批次
            start = int((i * BATCH_SIZE) % data_num)
            end = min(start + BATCH_SIZE, data_num)
            
            _ = sess.run(train_step,
                         feed_dict={x: X_train[start:end], y_: Y_train[start:end]})
            
            train_loss = sess.run(loss,
                                  feed_dict={x: X_train[start:end], y_: Y_train[start:end]})
            
            test_start = int(data_num * (1-test_rate))
            
            if start >= test_start:
                test_loss = sess.run(loss,
                                     feed_dict={x: X_test[start - test_start:end - test_start],
                                                y_: Y_test[start - test_start:end - test_start]})
            
            if i % 100 == 0:
                if start >= test_start:
                    print('snr：%d, '
                          'epoch %d, '
                          'train %d, '
                          'error_num %d, '
                          '训练集损失%.12f,'
                          '测试集损失%.12f' % (snr*10,
                                          epoch,
                                          i,
                                          sess.run(error_num, feed_dict={x: X_test[start - test_start:end - test_start],
                                                                         y_: Y_test[start-test_start:end-test_start]}),
                                          train_loss,
                                          test_loss))
                    train_loss_snr.append(train_loss)
                    test_loss_snr.append(test_loss)
                else:
                    print('snr：%d,'
                          'epoch %d, '
                          'train %d, '
                          'error_num %d, '
                          '训练集损失%.12f' % (snr*10,
                                          epoch,
                                          i,
                                          sess.run(error_num, feed_dict={x: X_train[start:end], y_: Y_train[start:end]}),
                                          train_loss))
                    train_loss_snr.append(train_loss)
    time_end = time.time()
    print('训练一个snr所用时间:%.2f'%(time_end-time_start))
    del(X_train, X_test, Y_train, Y_test)
    time_list.append(time_end - time_start)
    train_loss_list.append(train_loss_snr)
    test_loss_list.append(test_loss_snr)
    

print('snr:', SNR)
print('loss:', snr_ber)

plt.figure(figsize=(10, 10))
plt.plot(SNR, snr_ber)
plt.grid()
plt.xlabel('SNR')
plt.ylabel('error_rate')
plt.title('Deep Learning on channel estimate')
plt.savefig('./error_rate' + '%s'%(time.strftime('%d_%H_%M')) + '.jpg')
plt.show()

pd.DataFrame(data=train_loss_list, index=SNR).T.plot()
plt.title('Train loss')
plt.grid()
plt.xlabel('SNR')
plt.xticks(SNR)
plt.ylabel('loss')
plt.savefig('./train_loss' + '%s'%(time.strftime('%d_%H_%M')) + '.jpg')
plt.show()

pd.DataFrame(data=test_loss_list, index=SNR).T.plot()
plt.title('Test loss')
plt.grid()
plt.xlabel('SNR')
plt.xticks(SNR)
plt.ylabel('loss')
plt.savefig('./test_loss' + '%s'%(time.strftime('%d_%H_%M')) + '.jpg')
plt.show()

pd.Series(data=time_list, index=SNR).plot()
plt.title('time using')
plt.grid()
plt.xlabel('SNR')
plt.ylabel('time/s')
plt.savefig('./time_use' + '%s'%(time.strftime('%d_%H_%M')) + '.jpg')
plt.show()
