import scipy.io as sio
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(snr, test_rate):
    # 读取数据
    input_data = pd.read_csv(os.path.join('data', 'input%d.csv' % snr), header=None)

    output_data = pd.read_csv(os.path.join('data', 'output%d.csv' % snr), header=None)

    x_train, x_test, y_train, y_test = train_test_split(input_data.T,
                                                        output_data.T,
                                                        test_size=test_rate)

    return x_train.values, x_test.values, y_train.values, y_test.values


def read_data_undistortion(snr, test_rate):
    real = pd.DataFrame(sio.loadmat(os.path.join('非失真数据', 'Real_%d.mat' % snr)).get('Real_%d' % snr).T)
    imag = pd.DataFrame(sio.loadmat(os.path.join('非失真数据', 'Imag_%d.mat' % snr)).get('Imag_%d' % snr).T)
    Y = sio.loadmat(os.path.join('非失真数据', 'Test_label%d.mat' % snr)).get('Test_label%d' % snr).T

    X = pd.concat([real, imag], axis=1)  # 把实部和虚部放一起

    x_train, x_test, y_train, y_test = train_test_split(X.values,
                                                        Y,
                                                        test_size=test_rate)

    return x_train.values, x_test.values, y_train.values, y_test.values
