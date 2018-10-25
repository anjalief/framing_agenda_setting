#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models import Word2Vec, KeyedVectors
import sys
sys.path.append("..")
from article_utils import SentenceIter
from data_iters import BackgroundIter
import glob
from params import Params
import os

params = Params()

# Use this when we only train embeddings with target corpus
NYT_ONLY=True

class MFC_NYT_iter(object):
    def __init__(self, nyt_iter, mfc_iter):
        self.nyt_iter = nyt_iter
        self.mfc_iter = mfc_iter

    def __iter__(self):
        for sentence in self.nyt_iter:
            yield sentence
        for sentence in self.mfc_iter:
            yield sentence

def main():
    article_glob = params.NYT_PATH


    sentence_iter = SentenceIter(article_glob, verbose=False, skip_corrections=True)

    if NYT_ONLY:
        base_model = Word2Vec(sentence_iter, size=200, window=5, min_count=100, workers=10)
        fp = open(params.ENGLISH_BASE_MODEL, "wb")
    else:
        mfc_files = os.path.join(params.MFC_RAW_PATH, "*/json/*.json")
        mfc_glob = glob.iglob(mfc_files)
        mfc_iter = BackgroundIter(mfc_glob)
        dual_iter = MFC_NYT_iter(sentence_iter, mfc_iter)
        base_model = Word2Vec(dual_iter, size=200, window=5, min_count=100, workers=10)
        fp = open(params.ENGLISH_MFC_MODEL, "wb")

    base_model.save(fp)
    fp.close()

if __name__ == "__main__":
    main()
