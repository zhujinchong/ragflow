# -*- coding: utf-8 -*-

import copy
import json
import logging
import math
import re

import numpy as np
from elasticsearch_dsl import Q
from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity

from rag.nlp import rag_tokenizer, term_weight, synonym
from rag.nlp.sub_special_char import rm_stop_words
"""
1、组装es查询语句
2、计算query和chunk相似度
"""
class EsQueryer:
    def __init__(self, es):
        self.es = es
        # self.flds = ["ask_tks^10", "ask_small_tks"]
        self.flds = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "content_ltks^2",
            "content_sm_ltks"]
        # 权重
        self.term_weight_dealer = term_weight.Dealer()
        # 同义词
        self.synonym_dealer = synonym.Dealer()

    @staticmethod
    def subSpecialChar(line):
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|~\^])", r"\\\1", line).strip()

    def _isChinese(self, line):
        arr = re.split(r"[ \t]+", line)
        if len(arr) <= 3:
            return True
        e = 0
        for t in arr:
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1
        return e * 1. / len(arr) >= 0.7

    @staticmethod
    def rmWWW(txt):
        patts = [
            (
                r"是*(什么样的|哪家|一下|那家|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀)是*",
                ""),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down) ",
                " ")
        ]
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        return txt

    def question(self, txt, min_match="60%"):
        txt = rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower()))
        txt = rm_stop_words(txt)
        print("question(rm_stop_words): " + txt)
        if not self._isChinese(txt):
            tks = rag_tokenizer.tokenize(txt).split(" ")
            tks_w = self.term_weight_dealer.weights(tks)
            q = ["{}^{:.4f}".format(tk, w) for tk, w in tks_w if tk]
            # 相邻token组合
            for i in range(1, len(tks_w)):
                q.append("\"%s %s\"^%.4f" % (tks_w[i - 1][0], tks_w[i][0], max(tks_w[i - 1][1], tks_w[i][1]) * 2))
            # 如果q是空，直接检索输入字符串
            if not q:
                q.append(txt)
            return Q("bool",
                     must=Q("query_string", fields=copy.deepcopy(self.flds),
                            type="best_fields", query=" ".join(q),
                            boost=1)
                     ), tks

        def need_fine_grained_tokenize(tk):
            if len(tk) < 4:
                return False
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False
            return True

        qs, keywords = [], []
        for tt in rag_tokenizer.tokenize(txt).split(" ")[:256]:
            if not tt:
                continue
            twts = self.term_weight_dealer.weights([tt])
            syns = self.synonym_dealer.lookup(tt)
            logging.info(json.dumps(twts, ensure_ascii=False))
            tms = []
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                keywords.append(tk)

                sm = rag_tokenizer.fine_grained_tokenize(tk).split(" ") if need_fine_grained_tokenize(tk) else []
                sm = [m for m in sm if len(m) > 1]
                if len(sm) < 2:
                    sm = []
                tk_syns = self.synonym_dealer.lookup(tk)
                if tk.find(" ") > 0:
                    tk = f'\"{tk}\"'
                if tk_syns:
                    tk = f'({tk} {" ".join(tk_syns)})'
                if sm:
                    tk = f'{tk} OR \"{" ".join(sm)}\" OR (\"{" ".join(sm)}\"~2)^0.5'
                tms.append((tk, w))
            tms = " ".join([f"({t})^{w}" for t, w in tms])
            if len(twts) > 1:
                tms += f' (\"{" ".join([t for t, _ in twts])}\"~4)^1.5'
            if re.match(r"[0-9a-z ]+$", tt):
                tms = f'(\"{tt}\" OR \"{rag_tokenizer.tokenize(tt)}\")'
            if not tms:
                tms = tt
            tms = f"({tms})^5"
            if syns:
                syns = " OR ".join([f'\"{rag_tokenizer.tokenize(s)}\"^0.7' for s in syns])
                tms += f" OR ({syns})^0.7"
            qs.append(tms)

        mst = []
        # 如果q是空，直接检索输入字符串
        if qs:
            mst.append(
                Q("query_string", fields=copy.deepcopy(self.flds), type="best_fields",
                  query=" OR ".join([f"({t})" for t in qs if t]), boost=1, minimum_should_match=min_match)
            )

        return Q("bool",
                 must=mst,
                 ), keywords

    # 混合相似度分数计算：计算 query 和 [chunk] 之间的相似度
    # 返回：混合相似度分数，分词分数，向量分数
    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        sims = CosineSimilarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    # 计算 atks 和 btkss 之间的相似度分数
    # atks: tokenize(回答的一句话); btkss: [tokenize(chunk)，]
    # atks: [keyword, ]; btkss: [content_ltks + title_tks + important_kwd, ]
    def token_similarity(self, atks, btkss):
        def toDict(tks):
            """"{token:weight}"""
            d = {}
            if isinstance(tks, str):
                tks = tks.split(" ")
            for t, c in self.term_weight_dealer.weights(tks):
                if t not in d:
                    d[t] = 0
                d[t] += c
            return d

        atks = toDict(atks)
        btkss = [toDict(tks) for tks in btkss]
        return [self._similarity(atks, btks) for btks in btkss]

    # 计算 两个句子（关键字+权重） 之间的相似度分数
    def _similarity(self, qtwt, dtwt):
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v
        q = 1e-9
        for k, v in qtwt.items():
            q += v  # * v
        return s / q / max(1, math.sqrt(math.log10(max(len(qtwt.keys()), len(dtwt.keys())))))
