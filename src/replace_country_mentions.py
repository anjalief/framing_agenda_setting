#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
sys.path.append("..")
sys.path.append("../diachronic_embeddings")
from article_utils import LoadArticles, get_year_month, NEW_ARTICLE_TOKEN
from utils import morph_stem
import argparse
import multiprocessing
import os
import glob

pattern = None
output_dir = None
reg_subs={}
lower = True
stem = True

test_text = "Здравоохранение : DGDGDG миллиардов африканской Намибии на улучшение медицинского обслуживания . Буш против абортов и клониСШАрования человека . На борьбу со СПИДом в Африке США выделят DGDG миллиардов долларов ."

def parse_subs_file(filename):
    bad_to_good = {}
    for line in open(filename).readlines():
        splits = line.strip().split(",")
        # first word in the line is the one we want to use
        for bad in splits[1:]:
            if bad.strip() == "":
                continue
            # I think we only want to sub full words
            # text has been tokenized
            key = bad.strip().lower()
            val = splits[0].strip()
            if stem:
                key = morph_stem(key)
            bad_to_good[key] = val
    print (bad_to_good)
    return bad_to_good

def parse_liwc_file(filename):
    bad_to_good = {}
    count_sep = 0 # we start getting words after we see 2 %
    for line in open(filename).readlines():
        while count_sep < 2:
            if "%" in line:
                count_sep += 1
            continue
        splits = line.split()
        # add a space because we want words that start with this
        # text is all tokenized
        # add trailing space because we want to sub entire word
        # then need to add back space
        bad_to_good[splits[0].lower()] =  "_".join(splits[1:])
    print(bad_to_good)
    return bad_to_good

def do_sub(article_name):
    articles, article_indices = LoadArticles(article_name)

    # we can always put this in places later
    filename = os.path.basename(article_name)
    new_filepath = os.path.join(output_dir, filename)

    fp = open(new_filepath, "a+")

    for text in articles:
        text = text.replace("\n", " ").replace("\r", " ")
        if lower:
            text = text.lower()
        if stem:
            words = text.split()
            new_words = []
            for w in words:
                new_words.append(morph_stem(w))
            text = " ".join(new_words)
        text = pattern.sub(lambda m: reg_subs[re.escape(m.group(0))], text)

        # write article
        fp.write(NEW_ARTICLE_TOKEN + "\n")
        fp.write(text)
        fp.write("\n\n")

    fp.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--article_glob')
    parser.add_argument('--subs_file', default="cleaned.txt")
    parser.add_argument('--outpath')
    args = parser.parse_args()

    global reg_subs
    if "LIWC" in args.subs_file:
        bad_to_good = parse_liwc_file(args.subs_file)
        global lower
        lower = True
    else:
        bad_to_good = parse_subs_file(args.subs_file)
    reg_subs = dict((re.escape(k), v) for k, v in bad_to_good.items())

    global pattern
    pattern = re.compile(r'\b(' + '|'.join(reg_subs.keys()) + r')\b')

    global output_dir
    output_dir = args.outpath

    file_names = glob.iglob(args.article_glob)

    # parallelize over files, not articles so that we're only writing
    # 1 file at a time

    # global test_text
    # test_text = test_text.replace("\n", " ").replace("\r", " ")
    # test_text = test_text.lower()
    # text = pattern.sub(lambda m: reg_subs[re.escape(m.group(0))], test_text)
    # print(test_text)
    # print(text)

    pool = multiprocessing.Pool(processes=8)
    out_data = pool.map(do_sub, file_names)


if __name__ == "__main__":
    main()
