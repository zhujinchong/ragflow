# -*- coding: utf-8 -*-
import os

import numpy as np
from jieba import analyse

from api.utils.file_utils import get_project_base_directory

"""
[（关键词，权重）]
"""


class Dealer:
    def __init__(self):
        try:
            # 停用词表
            # analyse.set_stop_words(os.path.join(get_project_base_directory(), "rag/res/stop_words.txt"))
            a = 1
            # 逆向文件频率（IDF）文本语料库
            # analyse.set_idf_path(os.path.join(file_dir, "idf.txt.big"))
        except Exception as e:
            print("[WARNING] Load stop_words.txt FAIL!")

    def weights(self, token_list):
        """
        输入必须是list,但内容可以是：字符串、token
        """
        tw = []
        for t in token_list:
            tw.extend(analyse.extract_tags(t, withWeight=True))
        S = np.sum([s for _, s in tw])
        return [(t, s / S) for t, s in tw]


# if __name__ == '__main__':
#     dl = Dealer()
#     from rag_tokenizer import tokenize
#     s = '  your name tom cat if not wh your real name'
#     print(dl.weights([s]))
    # token_list = tokenize(s)
    # print(token_list)
    # print(dl.weights([token_list]))
    # print(dl.weights(token_list.split(" ")))
#
