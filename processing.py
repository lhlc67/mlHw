import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from scipy.signal import savgol_filter
from copy import deepcopy
import pywt


def PlotSpectrum(spec, title='原始光谱', x=0, m=5):
    """
    :param spec: shape (n_samples, n_features)
    :return: plt
    """
    if isinstance(spec, pd.DataFrame):
        spec = spec.values
    spec = spec[:, :(spec.shape[1]-1)]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    wl = np.linspace(x, x+(spec.shape[1]-1)*m, spec.shape[1])
    with plt.style.context(('ggplot')):
        fonts = 6
        plt.figure(figsize=(5.2, 3.1), dpi=200)
        plt.plot(wl, spec.T)
        plt.xlabel('Wavelength (nm)', fontsize=fonts)
        plt.ylabel('reabsorbance', fontsize=fonts)
        plt.title(title, fontsize=fonts)
    return plt


def D1(data):
    """ 一阶中心差分 """
    zero_col = np.zeros(data.shape[0])

    # 前后都加上一列0
    fill_zero = np.column_stack(
        (np.insert(data, 0, zero_col, axis=1), zero_col))

    # 一阶中心差分
    d1 = np.gradient(fill_zero, axis=1)

    # 去掉插入的0的两列
    res = d1[:, 1:-1]

    return res


def D2(data):
    """ 二阶中心差分 """
    res_d2 = D1(D1(data))
    return res_d2

def DD3(data):
    data = deepcopy(data)
    temp1 = np.mean(data, axis=0)
    temp2 = np.tile(temp1, data.shape[0]).reshape(
        (data.shape[0], data.shape[1]))
    return data - temp2

def mean_centralization(data):
    """
    均值中心化
    """
    data = deepcopy(data)
    temp1 = np.mean(data, axis=0)
    temp2 = np.tile(temp1, data.shape[0]).reshape(
        (data.shape[0], data.shape[1]))
    return data - temp2



def standardlize(sdata):
    """
    标准化
    """
    sdata = deepcopy(sdata)
    if isinstance(sdata, pd.DataFrame):
        sdata = sdata.values

    sdata = preprocessing.scale(sdata)
    return sdata


def msc(sdata):

    sdata = deepcopy(sdata)
    if isinstance(sdata, pd.DataFrame):
        sdata = sdata.values

    n = sdata.shape[0]  # 样本数量
    k = np.zeros(sdata.shape[0])
    b = np.zeros(sdata.shape[0])

    M = np.array(np.mean(sdata, axis=0))

    from sklearn.linear_model import LinearRegression

    for i in range(n):
        y = sdata[i, :]
        y = y.reshape(-1, 1)
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_

    spec_msc = np.zeros_like(sdata)
    for i in range(n):
        bb = np.repeat(b[i], sdata.shape[1])
        kk = np.repeat(k[i], sdata.shape[1])
        temp = (sdata[i, :] - bb) / kk
        spec_msc[i, :] = temp
    return spec_msc


def snv(sdata):
    """
    标准正态变量变换
    """
    sdata = deepcopy(sdata)
    if isinstance(sdata, pd.DataFrame):
        sdata = sdata.values
    temp1 = np.mean(sdata, axis=1)   # 求均值
    temp2 = np.tile(temp1, sdata.shape[1]).reshape((sdata.shape[0],
                                                    sdata.shape[1]), order='F')
    temp3 = np.std(sdata, axis=1)    # 标准差
    temp4 = np.tile(temp3, sdata.shape[1]).reshape((sdata.shape[0],
                                                    sdata.shape[1]), order='F')
    return (sdata - temp2) / temp4


def max_min_normalization(data):
    """
    最大最小归一化
    """
    data = deepcopy(data)
    # min = np.min(data, axis=0)
    # max = np.max(data, axis=0)
    # res = (data - min) / (max - min)
    if isinstance(data, pd.DataFrame):
        data = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    res = min_max_scaler.fit_transform(data.T)
    return res.T


def vector_normalization(data):
    """
    矢量归一化
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    x_mean = np.mean(data, axis=1)   # 求均值
    x_means = np.tile(x_mean, data.shape[1]).reshape(
        (data.shape[0], data.shape[1]), order='F')
    x_2 = np.power(data, 2)
    x_sum = np.sum(x_2, axis=1)
    x_sqrt = np.sqrt(x_sum)
    x_low = np.tile(x_sqrt, data.shape[1]).reshape(
        (data.shape[0], data.shape[1]), order='F')
    return (data - x_means) / x_low


def SG(data, w=5, p=3, d=0):
    """
    SG平滑 
    待处理
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    # data_sg = []
    # for item in data.iterrows():
    #     # print(item[0], item[1])
    #     data_sg.append(savgol_filter(item[1], x, y, mode=mode))
    # return DataFrame(data_sg, columns=absorbances)
    # savgol_filter(X, 2 * w + 1, polyorder=p, deriv=0)
    data = savgol_filter(data, w, polyorder=p, deriv=d)
    return data


def wave(data_x):  # 小波变换
    data_x = deepcopy(data_x)
    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.values

    def wave_(data_x):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data_x), w.dec_len)
        coeffs = pywt.wavedec(data_x, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data_x.shape[0]):
        if (i == 0):
            tmp = wave_(data_x[i])
        else:
            tmp = np.vstack((tmp, wave_(data_x[i])))
    return tmp


def move_avg(data_x, n=15, mode="valid"):
    # 滑动平均滤波
    data_x = deepcopy(data_x)
    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.values
    tmp = None
    for i in range(data_x.shape[0]):
        if (i == 0):
            tmp = np.convolve(data_x[i, :], np.ones((n,)) / n, mode=mode)
        else:
            tmp = np.vstack(
                (tmp, np.convolve(data_x[i, :], np.ones((n,)) / n, mode=mode)))
    return tmp
