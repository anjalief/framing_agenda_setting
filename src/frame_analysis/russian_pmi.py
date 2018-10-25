# This script is for measuring salience of frames in test
# (i.e. Russian) articles. Frames should be created with parse_frames.py

import sys
sys.path.append("..")
from article_utils import *

import argparse
import pickle
from collections import Counter, defaultdict
from params import Params

keywords = ["USA"]

def ComputeFramePMI(frame_to_lex, articles, params, raw_freq = False):
    frame_to_article_count = defaultdict(int)
    frame_to_article_and_usa_count = defaultdict(int)
    usa_count = 0
    total_article_count = 0

    for article in articles:
        total_article_count += 1
        words = Counter(article.split())
        if sum([words[k] for k in keywords]) >= 2:
            usa_count += 1
        for f in frame_to_lex:
            lex = frame_to_lex[f]
            count_frame_words = sum([words[w] for w in lex])
            if count_frame_words >= params.LEX_COUNT:
                frame_to_article_count[f] += 1
                if sum([words[k] for k in keywords]) >= 2:
                    frame_to_article_and_usa_count[f] += 1

    frame_to_pmi = {}
    total_article_count = float(total_article_count)
    for f in frame_to_lex:
        if raw_freq:
            frame_to_pmi[f] = frame_to_article_and_usa_count[f] / usa_count
            continue
        num = (frame_to_article_and_usa_count[f] / total_article_count)
        denom = (frame_to_article_count[f] / total_article_count) * (usa_count / total_article_count)
        if (denom == 0):
            frame_to_pmi[f] = 0
            continue
        # This is NORMALIZED PMI
        frame_to_pmi[f] = math.log(num / denom, 2) / -math.log(num, 2)
    return frame_to_pmi

def do_stuff(date_seq, filenames, frame_to_lex, whole_corpus, params):
    print(" ", end=";")
    for f in frame_to_lex:
        print (f, end=";")
    print("")

    if whole_corpus:
        all_articles = []
        for date,filename in zip(date_seq,filenames):
            assert(len(filename) == 1)
            articles,_ = LoadArticles(filename[0], verbose=False)
            all_articles += articles
        frame_to_pmi = ComputeFramePMI(frame_to_lex, all_articles, params)
        for f in frame_to_lex:
            print (frame_to_pmi[f], end=";")
        print("")
        return


    for date,filename in zip(date_seq,filenames):
        if len(filename) > 1:
            articles = []
            for f in filename:
                a,_ = LoadArticles(f, verbose=False)
                articles += a
        else:
            articles,_ = LoadArticles(filename[0], verbose=False)
        frame_to_pmi = ComputeFramePMI(frame_to_lex, articles, params)
        print(date, end=";")
        for f in frame_to_lex:
            print (frame_to_pmi[f], end=";")
        print("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/usr1/home/anjalief/corpora/russian/yearly_mod_subs/iz_lower/init_files/")
    parser.add_argument("--whole_corpus", action='store_true')
    parser.add_argument("--frame_lex", default="./cache/frame_to_lex_v2_200.pickle")
    parser.add_argument('--timestep', type=str,
                        default='monthly',
                        choices=['monthly', 'quarterly', 'semi', 'yearly'])
    args = parser.parse_args()

    params = Params()

    date_seq, filenames = get_files_by_time_slice(args.input_path, args.timestep)
    frame_to_lex = pickle.load(open(args.frame_lex, "rb"))

    do_stuff(date_seq, filenames, frame_to_lex, args.whole_corpus, params)


if __name__ == "__main__":
    main()
