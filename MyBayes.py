# @Time : 2020/11/28 11:04 
# @Author : huzhensha
# @Email : 1292776129@qq.com
# @File : MyBayes.py
# @Software: PyCharm

import numpy as np
import scipy.sparse as ss


class MyNaiveBayes(object):
    def __init__(self):
        self.vocabulary = []  # 词典
        self.idf = []  # 词典的IDF权重向量  逆文档频率
        self.tf = []  # 训练集的权值矩阵  词频
        self.tdm = []  # P(x|yi)
        self.p_cate = {}  # P(y_i)是一个类别字典
        self.labels = []  # 对应每个文本的分类，是一个外部导入的列表
        self.doc_length = 0  # 训练集文本数
        self.vocab_len = 0  # 词典词长
        self.test_list = []  # 测试集

    def fit(self, train_set, train_label, voculabry):
        """
        导入和训练数据集，生成算法必须的参数和数据结构
        :param train_set:
        :param train_label:
        :return:
        """
        self.cate_prob(train_label)  # 计算每个分类在数据集中的概率P(y_i)
        self.doc_length = len(train_set)

        # d = {}
        # for doc in train_set:
        #     for word in doc:
        #         if word in d:
        #             d[word] = d[word] + 1  # 生成词典
        #         else:
        #             d[word] = 1
        # after = list(sorted(d.items(), key=lambda item: item[1], reverse=True))
        # required_cnt = 10000
        # cnt = 0
        # for key in after.items():
        #     cnt += 1
        #     if cnt > required_cnt:
        #         break
        #     self.vocabulary.append(key)
        self.vocabulary = voculabry
        self.vocab_len = len(self.vocabulary)
        self.calc_tfidf_tdm(train_set)
        # self.build_tdm()  # 按分类累计向量空间的每维值P(x|y_i)

    def cate_prob(self, class_vec):
        """
        计算在数据集中每个分类的概率P(y_i)
        :param class_vec:
        :return:
        """
        self.labels = class_vec
        for temp in set(self.labels):  # 获取全部分类
            self.p_cate[temp] = float(self.labels.count(temp)) / float(len(self.labels))

    def calc_tfidf_tdm(self, train_set):
        """
        计算 tf-idf
        :param train_set:
        :return:
        """
        self.idf = np.ones([1, self.vocab_len])  # 填充1,防止零除,拉普拉斯平滑(防止新词出现)
        self.tdm = np.zeros([len(self.p_cate), self.vocab_len])
        for doc_index in range(self.doc_length):
            self.tf = np.zeros([1, self.vocab_len])
            for word in train_set[doc_index]:
                if word in self.vocabulary:
                    self.tf[0, self.vocabulary.index(word)] += 1
                    self.idf[0, self.vocabulary.index(word)] += 1  # 统计出现该单词的文本数量
            # 消除不同句长导致的偏差
            self.tf[0] = self.tf[0] / float(len(train_set[doc_index]) + 1)  # 防止空白文章出现0/0
            self.tdm[self.labels[doc_index]] += self.tf[0]  # 将同一类别的词向量空间值加总
        self.idf = np.log(float(self.doc_length) / self.idf)
        self.tdm = np.multiply(self.tdm, self.idf)  # 矩阵与向量的点乘 TFxIDF
        sum_list = np.zeros([len(self.p_cate), 1])  # 统计每个分类的所有词的总权重
        for cate_index in range(len(self.p_cate)):
            sum_list[cate_index] = np.sum(self.tdm[cate_index])  # 类别cate_index的所有词的总权重
        self.tdm = self.tdm / sum_list  # 生成P(x|y_i) 类别yi情况下出现各个词的概率

    # def build_tdm(self):
    #     """
    #     按分类累计计算向量空间的每维值P(x|y_i)
    #     :return:
    #     """
    #     self.tdm = np.zeros([len(self.p_cate), self.vocab_len])  # 类别行x词典列
    #     sum_list = np.zeros([len(self.p_cate), 1])  # 统计每个分类的总值
    #     for doc_index in range(self.doc_length):
    #         # 将同一类别的词向量空间值加总
    #         self.tdm[self.labels[doc_index]] += self.tf[doc_index]
    #         # 统计每个分类的总值——是一个标量
    #         sum_list[self.labels[doc_index]] = np.sum(self.tdm[self.labels[doc_index]])  # 类别yi的所有词的总权重
    #     self.tdm = self.tdm / sum_list  # 生成P(x|y_i) 类别yi情况下出现各个词的概率

    def test_data(self, test_data):
        """
        将测试集映射到当前词典
        :param test_data:
        :return:
        """
        self.test_list = np.zeros([len(test_data), self.vocab_len])  # 生成行数为测试机文档数，且列数为词长的矩阵
        for doc_index in range(len(test_data)):
            for word in test_data[doc_index]:
                if word in self.vocabulary:
                    self.test_list[doc_index, self.vocabulary.index(word)] += 1

    def predict(self, test_list):
        """
        预测分类结果，输出预测的分类类别
        :param test_list:
        :return:
        """
        res_pred = np.zeros(len(test_list))  # 存储每个文档的预测分类
        for doc_index in range(len(test_list)):  # 预测每一篇doc
            test_tf = np.zeros(self.vocab_len)  # 统计词频
            for word in test_list[doc_index]:
                if word in self.vocabulary:
                    test_tf[self.vocabulary.index(word)] += 1
            p_pred = -1  # 初始化类别概率
            type_pred = -1  # 初始化类别名称
            for tdm_p, cate in zip(self.tdm, self.p_cate):  # zip创建二元组
                # P(x|y_i) P(y_i)
                # 变量tdm，计算最大分类值
                # tdm_p 为P(x|Ci)  p_cate[cate]为P(Ci)   Ci为当前类别
                # sum对X所有维度计算的结果求和
                # temp = np.sum(test_list[doc_index] * tdm_p * self.p_cate[cate])  # temp为贝叶斯公式中的分子 P(x|Ci) * P(Ci)
                temp = 0
                for i in range(len(tdm_p)):
                    temp += np.log(test_tf[i] * tdm_p[i] + 1)  # 防止零概率，加1取对数
                temp *= self.p_cate[cate]
                if temp > p_pred:
                    p_pred = temp
                    type_pred = cate
            res_pred[doc_index] = type_pred
        return res_pred
