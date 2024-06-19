# -*- coding: utf-8 -*-

import os
import re
import sys

import jieba
from hanziconv import HanziConv
from jieba import posseg
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from api.utils.file_utils import get_project_base_directory


class RagTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        try:
            jieba.load_userdict(os.path.join(get_project_base_directory(), "rag/res/user_dict.txt"))
        except Exception as e:
            print("[WARNING]: Load userdict.txt FAIL!", file=sys.stderr)

    def _strQ2B(self, ustring):
        """全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        """繁体转简体"""
        return HanziConv.toSimplified(line)

    def tag(self, tk):
        words = posseg.lcut(tk)
        if len(words) != 1:
            return ""
        return words[0].flag

    def tokenize(self, line):
        line = self._strQ2B(line).lower()
        line = self._tradi2simp(line)
        zh_num = len([1 for c in line if is_chinese(c)])
        if zh_num == 0:
            return " ".join([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(line)])
        token_list = jieba.lcut(line, cut_all=False)
        token_list = [self.stemmer.stem(self.lemmatizer.lemmatize(t)) if re.match(r"[a-zA-Z_-]+$", t) else t for t in
                      token_list]
        token_list = [x.strip() for x in token_list if x.strip()]
        return " ".join(token_list)

    def fine_grained_tokenize(self, line):
        line = self._strQ2B(line).lower()
        line = self._tradi2simp(line)
        zh_num = len([1 for c in line if is_chinese(c)])
        if zh_num == 0:
            return " ".join([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(line)])
        token_list = jieba.lcut_for_search(line)
        token_list = [self.stemmer.stem(self.lemmatizer.lemmatize(t)) if re.match(r"[a-zA-Z_-]+$", t) else t for t in
                      token_list]
        token_list = [x.strip() for x in token_list if x.strip()]
        return " ".join(token_list)


def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B

# if __name__ == '__main__':
#     tt = RagTokenizer()
#     # print(tt.tag("天安门"))
#     line = "【天安门广场】是中国的@#$%大师傅。中新赛克是一家"
#     a = tt.tokenize(line)
#     print(a)
# print(jieba.lcut_for_search(a))
