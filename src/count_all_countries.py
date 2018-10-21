#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict, Counter
import os
import sys
from datetime import date
import pickle
from article_utils import *
from econ_utils import *
import glob
import math
import itertools

# Return number of country name words in timestep
# Number of articles that mention 1 country
# Number of articles that mention 2 countries
def count_mentions_in_articles(filenames, country_names):
    country_word_count = 0
    total_word_count = 0

    country_article_count = 0
    country_twice_count = 0
    country_thrice_count = 0
    total_article_count = 0
    articles_weighted_by_words = 0

    for filename in filenames:
        articles, _ = LoadArticles(filename, verbose=False)
        total_article_count += len(articles)
        for article in articles:
            countries_in_article = 0

            counter = Counter(article.split())
            total_word_count += sum([counter[q] for q in counter])
            for c in country_names:
                countries_in_article += counter[c]
                country_word_count += counter[c]
            if countries_in_article > 0:
                country_article_count += 1
            if countries_in_article > 1:
                country_twice_count += 1
                articles_weighted_by_words += len(article.split())
            if countries_in_article > 2:
                country_thrice_count += 1

    return country_article_count, country_twice_count, country_thrice_count, total_article_count, country_word_count, articles_weighted_by_words, total_word_count

def load_country_names(filename):
    names = []
    for line in open(filename).readlines():
        names.append(line.split(",")[0].strip())
    return names

def do_counts(files_grouped_by_date, subs_file):
    country_names = load_country_names(subs_file)

    # number of articles that mention country (normalize by number of articles)
    output_summary = []
    for filename in files_grouped_by_date:
        output_summary.append(count_mentions_in_articles(filename, country_names))
    return output_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help="Directory where data files are. Must include trailing backslash", default="../data/Izvestiia_processed/")
    parser.add_argument('--subs_file', help="File containing keywords to count", default="../data/usa.txt")
    parser.add_argument('--word_level', help="Print word-level counts instead of article-level counts", action='store_true')
    parser.add_argument('--timestep', type=str, help="specify what time increment to use for aggregating articles",
                        default='monthly',
                        choices=['monthly', 'quarterly', 'semi', 'yearly'])
    parser.add_argument("--econ_file", help="If you specify this, output will include compute correlation of counts with econ series (does NOT work for yearly)")
    args = parser.parse_args()

    date_seq, filenames = get_files_by_time_slice(args.input_path, args.timestep)
    output = do_counts(filenames, args.subs_file)

    assert(len(output) == len(date_seq))

    # Default is article level
    count_idx=1
    total_idx=3

    # Word level
    if args.word_level:
        count_idx=-3
        total_idx=-1

    for d,o in zip(date_seq, output):
        print (d, o[count_idx], o[total_idx], o[count_idx] / o[total_idx])

    # if we're missing econ data, this will just die
    if  args.econ_file:
        ratios = [o[count_idx] / o[total_idx] for o in output]

        # We skip 2016 for yearly
        if args.timestep == "yearly":
            ratios = ratios[:-1]
            date_seq = date_seq[:-1]

        econ_seq = load_econ_file(args.econ_file, args.timestep, date_seq)


        print(get_corr(econ_seq, ratios))

if __name__ == "__main__":
    main()
