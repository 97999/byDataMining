# @Time : 2020/11/28 15:49 
# @Author : huzhensha
# @Email : 1292776129@qq.com
# @File : getData.py 
# @Software: PyCharm

import Util as util
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

util.store_data('traintxt', "train")
util.store_data('testtxt', "test")


