# @Time : 2020/11/4 15:39 
# @Author : huzhensha
# @Email : 1292776129@qq.com
# @File : classification.py
# @Software: PyCharm

import os
import pickle

import jieba
import jieba.posseg
import time
import Util as util
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import LoadModelClassify as load_model


def fit_predict(classifier, x, y, x_test, y_test, name):
    """
    :param classifier: 分类器
    :param x: 特征训练数据
    :param y:  标签训练数据
    :param x_test:  特征测试数据
    :param y_test:  标签测试数据
    :return: 预测值， 准确率, 混淆矩阵, 算法消耗时间, 召回率
    """
    start_time = time.time()
    model = classifier
    model.fit(x, y)  # 训练
    joblib.dump(model, './cache/' + name + '.pkl')
    y_predict = model.predict(x_test)
    precision = metrics.accuracy_score(y_test, y_predict)  # 准确率 提取出的正确信息条数 /  提取出的信息条数
    matrix = metrics.confusion_matrix(y_test, y_predict)  # 混淆矩阵
    recall = metrics.classification_report(y_test, y_predict)  # 召回率 提取出的正确信息条数 /  样本中的信息条数
    # score = precision * recall * 2 / (precision + recall)
    end_time = time.time()
    consume_time = end_time - start_time
    return y_predict, precision, matrix, consume_time, recall


if os.path.exists('./cache/svm.pkl') \
        and os.path.exists('./cache/test_tfidf.pickle')\
        and os.path.exists('./cache/test_labels.pickle'):
    print("从已保存的模型加载")
    load_model.run()  # 从已经保存的模型加载
else:
    # 使用停止词
    stop_words = util.load_stop_words()

    # 读取数据
    # train_words_list, train_labels = util.load_file('D:/Code/Python/data/news')
    # test_words_list, test_labels = util.load_file('D:/Code/Python/data/test')
    # train_words_list, train_labels = util.load_file('text classification/train')
    # test_words_list, test_labels = util.load_file('text classification/test')
    train_words_list, train_labels = util.load_file('traintxt')
    test_words_list, test_labels = util.load_file('testtxt')

    # 对标签进行编码
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.fit_transform(test_labels)

    vectorizer = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, max_df=0.5, min_df=0.001)
    vectorizer.fit(train_words_list + test_words_list)
    train_tfidf = vectorizer.transform(train_words_list)
    test_tfidf = vectorizer.transform(test_words_list)
    pickle.dump(test_tfidf, open("./cache/test_tfidf.pickle", "wb"))  # 存储测试集及对应标签
    pickle.dump(test_labels, open("./cache/test_labels.pickle", "wb"))
    print(train_tfidf.shape)
    print(test_tfidf.shape)

    svc = svm.LinearSVC()
    result = fit_predict(svc, train_tfidf, train_labels, test_tfidf, test_labels, 'svm')
    print("支持向量机算法: ")
    for data in result:
        print(data)
