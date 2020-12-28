# @Time : 2020/11/26 23:40 
# @Author : huzhensha
# @Email : 1292776129@qq.com
# @File : LoadModelClassify.py 
# @Software: PyCharm
import pickle

import joblib
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import Util as util
import os
import time


def predict(model, x_test, y_test):
    start_time = time.time()
    y_predict = model.predict(x_test)
    precision = metrics.accuracy_score(y_test, y_predict)  # 准确率 提取出的正确信息条数 /  提取出的信息条数
    matrix = metrics.confusion_matrix(y_test, y_predict)  # 混淆矩阵
    recall = metrics.classification_report(y_test, y_predict)  # 召回率 提取出的正确信息条数 /  样本中的信息条数
    # score = precision * recall * 2 / (precision + recall)
    end_time = time.time()
    consume_time = end_time - start_time
    return y_predict, precision, matrix, consume_time, recall


def run():
    test_tfidf = pickle.load(open("./cache/test_tfidf.pickle", "rb"))
    test_labels = pickle.load(open("./cache/test_labels.pickle", "rb"))
    print(test_tfidf.shape)

    # SVM支持向量机
    path = './cache/svm.pkl'
    if os.path.exists(path):
        mnb = joblib.load(path)
        result = predict(mnb, test_tfidf, test_labels)
        print("支持向量机算法: ")
        for data in result:
            print(data)
