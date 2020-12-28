# @Time : 2020/11/30 0:28 
# @Author : huzhensha
# @Email : 1292776129@qq.com
# @File : start.py
# @Software: PyCharm

import os
import jieba
import jieba.posseg
from MyBayes import MyNaiveBayes
from .MyEncoder import MyEncoder
import Util as util

import joblib
import pickle
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


def cal_accuracy(label, result):
    num = 0
    for temp_label, temp_res in zip(label, result):
        if temp_res == temp_label:
            num += 1
    return num / len(label)


if os.path.exists('mycache1/mybayes.pickle'):
    print("从保存的模型中加载:")
    bayes = pickle.load(open("./mycache1/mybayes.pickle", "rb"))
    print("正在读取测试数据")
    test_data = []
    test_label = []
    test_data.extend(pickle.load(open('data_cache/test/auto.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_auto.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/cj.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_cj.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/cul.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_cul.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/fz.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_fz.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/mil.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_mil.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/sh.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_sh.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/stock.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_stock.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/tw.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_tw.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/ty.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_ty.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/yl.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_yl.pkl', 'rb')))

    # 对标签进行编码
    encoder = MyEncoder()
    encoder.fit(test_label)
    test_label = encoder.transform(test_label)

    print("正在预测")
    res = bayes.predict(test_data)


else:
    stop_words = util.load_stop_words()
    print("正在读取训练数据")
    data_list = []
    label_list = []
    data_list.extend(pickle.load(open('data_cache/train/auto.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_auto.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/cj.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_cj.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/cul.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_cul.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/fz.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_fz.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/mil.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_mil.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/sh.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_sh.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/stock.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_stock.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/tw.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_tw.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/ty.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_ty.pkl', 'rb')))
    data_list.extend(pickle.load(open('data_cache/train/yl.pkl', 'rb')))
    label_list.extend(pickle.load(open('data_cache/train/label_yl.pkl', 'rb')))

    print("正在读取测试数据")
    test_data = []
    test_label = []
    test_data.extend(pickle.load(open('data_cache/test/auto.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_auto.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/cj.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_cj.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/cul.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_cul.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/fz.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_fz.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/mil.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_mil.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/sh.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_sh.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/stock.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_stock.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/tw.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_tw.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/ty.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_ty.pkl', 'rb')))
    test_data.extend(pickle.load(open('data_cache/test/yl.pkl', 'rb')))
    test_label.extend(pickle.load(open('data_cache/test/label_yl.pkl', 'rb')))

    # 对标签进行编码
    encoder = MyEncoder()
    encoder.fit(label_list + test_label)
    label_list = encoder.transform(label_list)
    test_label = encoder.transform(test_label)

    print("正在训练模型")
    bayes = MyNaiveBayes()
    bayes.fit(data_list, label_list, pickle.load(open("./voca_cache/voca.pickle", "rb")))
    pickle.dump(bayes, open("./mycache1/mybayes.pickle", "wb"))
    print("正在预测")
    res = bayes.predict(test_data)

print("原分类依次为：")
print(test_label)
print(len(test_label))
print("朴素贝叶斯算法的结果为：")
print(len(res))
# print(cal_accuracy(test_label, res))
precision = metrics.accuracy_score(test_label, res)  # 准确率 提取出的正确信息条数 /  提取出的信息条数
print(metrics.confusion_matrix(test_label, res))  # 混淆矩阵
print(metrics.classification_report(test_label, res))  # 召回率 提取出的正确信息条数 /  样本中的信息条数
