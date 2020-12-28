# @Time : 2020/11/13 14:45 
# @Author : huzhensha
# @Email : 1292776129@qq.com
# @File : MyEncoder.py 
# @Software: PyCharm


class MyEncoder(object):
    def __init__(self):
        self.dictionary = {}

    def fit(self, data_set):
        labels = list(set(data_set))
        for index in range(len(labels)):
            self.dictionary[labels[index]] = index

    def transform(self, data):
        res = []
        for temp in data:
            res.append(self.dictionary[temp])
        return res
