# @Time : 2020/11/26 23:40
# @Author : huzhensha
# @Email : 1292776129@qq.com
# @File : LoadModelClassify.py
# @Software: PyCharm

import jieba
import jieba.posseg
import os
import joblib


def cut_words(file_path, stop_words=None):
    """
    对文本进行切词
    :param stop_words:
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    try:
        text = open(file_path, mode='r', encoding='utf-8').read()
    except:
        text = open(file_path, mode='r', encoding='gb18030').read()

    # 只取名词
    seg = jieba.posseg.cut(text)
    for w in seg:
        if w.flag == 'n' or w.flag == 'nr' or w.flag == 'ns':  # 名词，人名，地名
            if stop_words is None or w.word not in stop_words:
                text_with_spaces += w.word + ' '
    return text_with_spaces  # 返回字符串


def cut_words1(file_path, stop_words=None):
    """
    对文本进行切词
    :param stop_words:
    :param file_path: txt文本路径
    :return: 返回集合形式
    """
    text_list = []
    try:
        text = open(file_path, mode='r', encoding='utf-8').read()
    except:
        text = open(file_path, mode='r', encoding='gb18030').read()

    # 只取名词
    seg = jieba.posseg.cut(text)
    for w in seg:
        if w.flag == 'n' or w.flag == 'nr' or w.flag == 'ns':  # 名词，人名，地名
            if stop_words is None or w.word not in stop_words:
                text_list.append(w.word)
    return text_list  # 返回list


def load_file(file_dir, stop_words=None):
    """
    将路径下的所有文件加载
    :param stop_words:
    :param file_dir: txt文件目录
    :return: 分词后的文档列表和标签
    """
    words_list = []
    labels_list = []
    cate_list = os.listdir(file_dir)  # 文本集合
    for cate in cate_list:
        cate_path = file_dir + '/' + cate
        file_list = os.listdir(cate_path)
        for file in file_list:
            file_path = cate_path + '/' + file
            words_list.append(cut_words(file_path, stop_words))  # 每一个文件切分成带有空格标记的序列，然后拼接
            labels_list.append(cate)  # 标签也拼接在一起
    return words_list, labels_list


def load_file1(file_dir, stop_words=None):
    """
    将路径下的所有文件加载
    :param stop_words:
    :param file_dir: txt文件目录
    :return: 分词后的文档列表和标签
    """
    words_list = []
    labels_list = []
    cate_list = os.listdir(file_dir)  # 文本集合
    for cate in cate_list:
        cate_path = file_dir + '/' + cate
        file_list = os.listdir(cate_path)
        for file in file_list:
            file_path = cate_path + '/' + file
            words_list.append(cut_words1(file_path, stop_words))  # 每一个文件切分list，然后拼接
            labels_list.append(cate)  # 标签也拼接在一起

    return words_list, labels_list  # 数据格式为二维数组


def store_data(file_dir, name):
    """
    将路径下的所有文件加载
    :param stop_words:
    :param file_dir: txt文件目录
    :return: 分词后的文档列表和标签
    """

    cate_list = os.listdir(file_dir)  # 文本集合
    for cate in cate_list:
        words_list = []
        labels_list = []
        cate_path = file_dir + '/' + cate
        file_list = os.listdir(cate_path)
        for file in file_list:
            file_path = cate_path + '/' + file
            words_list.append(cut_words1(file_path))  # 每一个文件切分list，然后拼接
            labels_list.append(cate)  # 标签也拼接在一起
        joblib.dump(words_list, open("./data_cache/" + name + "/" + cate + ".pkl", "wb"))
        joblib.dump(labels_list, open("./data_cache/" + name + "/" + "label" + "_" + cate + ".pkl", "wb"))




def load_stop_words():
    stop_words = open('stopword.txt', 'r', encoding='utf-8').read()
    # 消去文本头部BOM标记(BOM是Unicode规范中推荐的标记字节顺序的方法)\ufeff
    stop_words = stop_words.encode('utf-8').decode('utf-8-sig')
    stop_words = stop_words.split('\n')  # 根据分隔符分隔
    return stop_words
