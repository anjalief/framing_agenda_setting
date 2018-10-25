#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models import Word2Vec, KeyedVectors
import argparse
from parse_frames import do_counts, words_to_pmi, seeds_to_real_lex, get_words_to_cut
from data_iters import FrameAnnotationsIter, BackgroundIter, get_sentence_level_test, load_json_as_list, load_codes, get_random_split, code_to_short_form, get_per_frame_split, FrameHardSoftIter
import os
from collections import Counter, defaultdict
import pickle
from random import shuffle
import operator
from scipy import spatial
import glob
from params import Params

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append("''")
stop_words.append('``')
stop_words.append('--')
stop_words.append("'s")
stop_words.append("n't")
stop_words.append("said")
import string

def get_top_words(input_file):
    dir_name = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
    base_name = os.path.join("cache", dir_name + ".counter")
    if (os.path.isfile(base_name)):
        return pickle.load(open(base_name, "rb"))

    c = Counter()
    num_articles = 0
    article_counter = Counter()
    for words in BackgroundIter(glob.iglob(input_file), sent_level=False):
        c.update(words)
        num_articles += 1
        article_counter.update(set(words))
    pickle.dump((c, num_articles, article_counter), open(base_name, "wb"))
    return c, num_articles, article_counter

# Theory that some frames we model better than others because they are less relavent to
# test text. Count how frequent each frame is in input text
def count_frames(code_to_str, frame_iter):
    frame_counter = Counter()
    text_count = 0
    for text,frames,_ in frame_iter:
        for frame in frames:
            frame_counter[frame] += 1
        text_count += 1

    for f in sorted(frame_counter):
        print(code_to_str[f], ";", frame_counter[f])

# NEW WAY -- train on top of NYT model
def get_wv_nyt_name(input_file, split_type, params):
    if split_type == "random" or split_type == 'kfold' or split_type == 'dcard' or split_type == 'dcard_tune':
        base_name = params.ENGLISH_MFC_MODEL
    else:
        base_name = os.path.join("cache", split_type + ".nyt.model")

    nyt_model = params.ENGLISH_BASE_MODEL

    if (os.path.isfile(base_name)):
        return base_name

    sentence_iter = BackgroundIter(glob.iglob(input_file), verbose=False)
    base_model = Word2Vec.load(nyt_model)
    count = 0
    for x in sentence_iter:
      count += 1
    base_model.train(sentence_iter, total_examples=count, epochs=base_model.epochs)

    fp = open(base_name, "wb")
    base_model.wv.save(fp)
    fp.close()

    return base_name

class frameTracker():
  def __init__(self):
    self.correct_positive = 0
    self.marked_positive = 0
    self.true_positive = 0
    self.marked_correct = 0

  def get_metrics(self, total):
    # return precision, recall, accuracy
    return self.correct_positive / max(float(self.marked_positive), 0.0001), \
           self.correct_positive / float(self.true_positive),   \
           self.marked_correct / float(total)

def test_primary_frame(code_to_lex, code_to_str, text_iter, do_print = True):
    total = 0
    correct = 0
    del code_to_lex[15.0] # Don't guess Other
    for text,_,true_frame in text_iter:
        if true_frame == "null" or true_frame is None:
            continue
        total += 1
        text_counter = Counter(text)

        sums = []
        for f in code_to_lex:
            sums.append((f, sum([text_counter[w] for w in code_to_lex[f]])))

        # we shuffle so that ties are randomly broken
        shuffle(sums)
        frame, word_count = max(sums, key=operator.itemgetter(1))
        # Mark as "Other" if it doesn't belong to any other frame
        if word_count < 4:
            frame = 15.0
        if frame == true_frame:
            correct += 1
    if do_print:
        print (float(correct) / float(total), total)
    return float(correct) / float(total)


# Find center of a set of vectors (unnormalized)
# by summing the vectors
def get_center(words, wv):
    embed_size = 0
    for w in words:
        if not w in wv:
            continue
        embed_size = len(wv[w])
        break
    center = [0 for i in range(0, embed_size)]

    for w in words:
        if not w in wv:
            continue
        center = [x+y for x,y in zip(center, wv[w])]
    return center

# First find center of context_words vectors, then return similarity between keyword and center
def get_mean_similarity(keywords, context_words, wv):
    context_center = get_center(context_words, wv)
    keywords_center = get_center(keywords, wv)
    return 1 - spatial.distance.cosine(context_center, keywords_center)

def test_primary_frame_wv(code_to_lex, code_to_str, text_iter, wv):
    total = 0
    correct = 0
    for text,_,true_frame in text_iter:
        if true_frame == "null" or true_frame is None:
            continue

        total += 1

        sums = []
        for f in code_to_lex:
            sums.append((f, get_mean_similarity(text, code_to_lex[f], wv)))

        # we shuffle so that ties are randomly broken
        shuffle(sums)
        frame, word_count = max(sums, key=operator.itemgetter(1))
        if frame == true_frame:
            correct += 1

    print (float(correct) / float(total), total)

def max_index(l):
    index, value = max(enumerate(l), key=operator.itemgetter(1))
    return str(index)


def test_sentence_annotations(code_to_lex, code_to_str, frame_to_contains, frame_to_doesnt):
    for f in sorted(code_to_lex):
        frame_tracker = frameTracker()
        total = 0

        for contains in frame_to_contains[f]:
            total += 1
            frame_tracker.true_positive += 1

            text_counter = Counter(contains)

            applies_frame = sum([text_counter[w] for w in code_to_lex[f]]) >= 1
            if applies_frame:
                frame_tracker.marked_correct += 1
                frame_tracker.marked_positive += 1
                frame_tracker.correct_positive += 1

        for doesnt in frame_to_doesnt[f]:
            total += 1
            text_counter = Counter(doesnt)
            applies_frame = sum([text_counter[w] for w in code_to_lex[f]]) >= 1
            if applies_frame:
                frame_tracker.marked_positive += 1
            else:
                frame_tracker.marked_correct += 1

        assert (frame_tracker.true_positive == len(frame_to_contains[f]))
        assert (total == len(frame_to_contains[f]) + len(frame_to_doesnt[f]))

        p,r,a = frame_tracker.get_metrics(total)
        if (p + r) == 0:
            print(code_to_str[f], "VERY BAD")
            continue
        print (code_to_str[f], ";",
           p, ";",
           r, ";",
           (2 * (p * r)/(p + r)), ";",
           a, ";")

def test_annotations(code_to_lex, code_to_str, frame_iter, lex_count=3, do_print=True):
  code_to_frame_tracker = {}
  for c in code_to_lex:
    code_to_frame_tracker[c] = frameTracker()

  total = 0
  for text,frames,_ in frame_iter:
    total += 1
    text_counter = Counter(text)

    for c in code_to_lex:
      applies_frame = (sum([text_counter[w] for w in code_to_lex[c]]) >= lex_count)

      gold_applies_frame = (c in frames)

      if applies_frame == gold_applies_frame:
        code_to_frame_tracker[c].marked_correct += 1

      if applies_frame:
        code_to_frame_tracker[c].marked_positive += 1
        if gold_applies_frame:
          code_to_frame_tracker[c].correct_positive += 1

      if gold_applies_frame:
        code_to_frame_tracker[c].true_positive += 1

  code_to_f1 = {}
  average_f1 = 0
  for c in sorted(code_to_frame_tracker):
    p,r,a = code_to_frame_tracker[c].get_metrics(total)
    if (p + r) == 0:
      code_to_f1[c] = 0
      continue

    code_to_f1[c] = (2 * (p * r)/(p + r))


    if do_print:
        print (code_to_str[c], ";",
               p, ";",
               r, ";",
               (2 * (p * r)/(p + r)), ";",
               a, ";")
        if code_to_str[c] == "Other":
            continue
        average_f1 += (2 * (p * r)/(p + r))
  if do_print:
      print ("AVERAGE", average_f1 / (len(code_to_frame_tracker) - 1))
  return code_to_f1

def test_hard_annotations(code_to_lex, code_to_str, frame_iter, lex_count=3):
  code_to_frame_tracker = {}
  for c in code_to_lex:
    code_to_frame_tracker[c] = frameTracker()

  total = 0
  for text,frame_to_all, frame_to_any in frame_iter:
    total += 1
    text_counter = Counter(text)

    for c in code_to_lex:
      applies_frame = (sum([text_counter[w] for w in code_to_lex[c]]) >= lex_count)

      # Check hard, it's only in doc if all annotators think it's in doc
      gold_applies_frame = frame_to_all[c]

      if applies_frame == gold_applies_frame:
        code_to_frame_tracker[c].marked_correct += 1

      if applies_frame:
        code_to_frame_tracker[c].marked_positive += 1
        if gold_applies_frame:
          code_to_frame_tracker[c].correct_positive += 1

      if gold_applies_frame:
        code_to_frame_tracker[c].true_positive += 1

  for c in sorted(code_to_frame_tracker):
    p,r,a = code_to_frame_tracker[c].get_metrics(total)
    if (p + r) == 0:
      print ("VERB BAD")
      return
    print (code_to_str[c], ";",
           p, ";",
           r, ";",
           (2 * (p * r)/(p + r)), ";",
           a, ";")

def get_data_split(split_type, params, frame = None):

  immigration = os.path.join(params.MFC_PATH, "immigration.json")
  tobacco = os.path.join(params.MFC_PATH, "tobacco.json")
  samsex = os.path.join(params.MFC_PATH, "samesex.json")
  full_background = os.path.join(params.MFC_RAW_PATH, "*/json/*.json")

  if split_type == 'tobacco':
      train_files = [immigration, samesex]
      test_files = tobacco
      test_background = os.path.join(params.MFC_RAW_PATH, "smoking/json/*.json")

  elif split_type == 'immigration':
      train_files = [tobacco, samesex]
      test_files = immigration
      test_background = os.path.join(params.MFC_RAW_PATH, "immigration/json/*.json")
  elif split_type == 'samesex':
      train_files = [tobacco, immigration]
      test_files = samesex
      test_background = os.path.join(params.MFC_RAW_PATH, "samesex/json/*.json")
  elif split_type == 'kfold':
      train_files = [tobacco, immigration, samesex]
      test_background = full_background
      assert(frame is not None)
      test_data, train_data = get_per_frame_split(train_files, frame)
      return train_data, test_data, test_background
  elif split_type == 'dcard':
      train_files = [immigration]
      test_background = full_background
      test_data, train_data = get_random_split(train_files, num_folds=10, filter_tone=True)
      return train_data, test_data, test_background
  elif split_type == 'dcard_tune':
      train_files = [immigration]
      test_background = full_background
      test_data, train_data = get_random_split(train_files, num_folds=50, filter_tone=True)
      return train_data, test_data, test_background
  else:
      assert (split_type == "random")
      # train_files = [tobacco, immigration, samesex]
      train_files = [immigration]
      test_background = full_background
      test_data, train_data = get_random_split(train_files)

      return train_data, test_data, test_background

  train_data = load_json_as_list(train_files)
  test_data = load_json_as_list([test_files])

  return train_data, test_data, test_background

def count_all_frames():
    immigration = os.path.join(params.MFC_PATH, "immigration.json")
    tobacco = os.path.join(params.MFC_PATH, "tobacco.json")
    samsex = os.path.join(params.MFC_PATH, "samesex.json")
    codes = os.path.join(params.MFC_PATH, "codes.json")

    train_files = [immigration, tobacco, samesex]

    code_to_str = load_codes(codes)
    train_data = load_json_as_list(train_files)
    doc_level_iter = FrameAnnotationsIter(train_data)
    count_frames(code_to_str, doc_level_iter)

def do_all(args, train_data, test_data, test_background, code_to_str, params, target_frame = None, do_print = True):
    wv_name = get_wv_nyt_name(test_background, args.split_type, params)

    corpus_counter, code_to_counter, word_to_article_count, total_article_count = do_counts(train_data)

    # Sometimes (kfold) we only care about 1 frame
    if target_frame is not None:
        code_to_counter = {f:code_to_counter[f] for f in [target_frame]}

    # cut infrequent words
    cut_words = get_words_to_cut(total_article_count, word_to_article_count, params.MIN_CUT, params.MAX_CUT)
    corpus_counter = Counter({c:corpus_counter[c] for c in corpus_counter if not c in cut_words})

    # calculate PMI
    corpus_count = sum([corpus_counter[k] for k in corpus_counter])
    code_to_lex = {}
    all_frames = set()
    for c in code_to_counter:
        if "primary" in code_to_str[c] or "headline" in code_to_str[c] or "primany" in code_to_str[c]:
            continue

        all_frames.add(c)
        # For the baseline, we just take 100 most frequent words
        if args.baseline:
            # remove stopwords
            code_to_counter[c] = Counter({w:code_to_counter[c][w] for w in code_to_counter[c] if w in corpus_counter and not w in stop_words and not w in string.punctuation})
            code_to_lex[c] = [q[0] for q in code_to_counter[c].most_common(100)]
        else:
            code_to_lex[c] = words_to_pmi(corpus_counter, corpus_count, code_to_counter[c], params.TO_RETURN_COUNT)
            # # Use same seeds as baseline
            # code_to_counter[c] = Counter({w:code_to_counter[c][w] for w in code_to_counter[c] if w in corpus_counter and not w in stop_words and not w in string.punctuation})
            # code_to_lex[c] = [q[0] for q in code_to_counter[c].most_common(100)]

    if do_print:
        print("*******************************************************************************")
        for c in code_to_lex:
            print (code_to_str[c], code_to_lex[c])
        print("*******************************************************************************")
        print("*******************************************************************************")

    top_words, num_articles, article_counter = get_top_words(test_background)
    vocab = sorted(top_words, key=top_words.get, reverse = True)[:params.VOCAB_SIZE]

    if args.baseline:
      code_to_new_lex = code_to_lex
    else:
      code_to_new_lex = {}
      for c in code_to_lex:
         code_to_new_lex[c] = seeds_to_real_lex(code_to_lex[c], wv_name, vocab, code_to_str[c], topn=params.VEC_SEARCH, threshold=params.SIM_THRESH)
#         code_to_new_lex[c] = code_to_lex[c] # Test no query expansions. This guy is pretty good

    # filter again, this time off of target corpus cause that's what we have to do in Russian
    words_to_cut = get_words_to_cut(num_articles, article_counter, params.MIN_CUT, params.MAX_CUT)
    for c in code_to_new_lex:
        code_to_new_lex[c] = [w for w in code_to_new_lex[c] if not w in words_to_cut]

    # make data iters
    doc_level_iter = FrameAnnotationsIter(test_data)
    short_codes = set([code_to_short_form(code) for code in code_to_str])
    hard_iter = FrameHardSoftIter(test_data, short_codes)
    # sentence level tests
    frame_to_contains, frame_to_doesnt = get_sentence_level_test(test_data, all_frames)

    # just return everything
    if not do_print:
        return code_to_new_lex, doc_level_iter


    for x in code_to_new_lex:
        print (code_to_str[x])
        print (code_to_new_lex[x])
    # print("*******************************************************************************")
    # print("Frame Counts;")
    # count_frames(code_to_str, doc_level_iter)
    print("*******************************************************************************")
    print("DOC")
    test_annotations(code_to_new_lex, code_to_str, doc_level_iter, lex_count=params.LEX_COUNT)
    # print("*******************************************************************************")
    # Skipping this for now
    # print("DOC HARD")
    # test_hard_annotations(code_to_new_lex, code_to_str, hard_iter)
    # print("*******************************************************************************")
    # print("SENTENCE")
    # test_sentence_annotations(code_to_new_lex,code_to_str, frame_to_contains, frame_to_doesnt)
    print("*******************************************************************************")
    print("PRIMARY")
    test_primary_frame(code_to_new_lex, code_to_str, doc_level_iter)
    print("*******************************************************************************")
    # # Real slow and doesn't work well
    # print("PRIMARY WV")
    # test_primary_frame_wv(code_to_new_lex, code_to_str, doc_level_iter, KeyedVectors.load(wv_name))

    to_save = {}
    for x in code_to_new_lex:
      to_save[code_to_str[x]] = code_to_new_lex[x]
    pickle.dump(to_save, open("cache/" + args.split_type + "_lex.pickle", "wb"))


    to_save = {}
    for x in code_to_lex:
      to_save[code_to_str[x]] = code_to_new_lex[x]
    pickle.dump(to_save, open("cache/" + args.split_type + "_base_lex.pickle", "wb"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action='store_true')
    # specify what to use as training data and what to use as test set. If random, hold out 20% of data of test
    # if kfold, we do a different data split for each frame, so that test and train data have same proportion
    # of the frame at the document level
    parser.add_argument("--split_type", choices=['tobacco', 'immigration', 'samesex', 'random', 'kfold', 'dcard', 'dcard_tune'])
    args = parser.parse_args()


    p = Params()

    code_file = os.path.join(p.MFC_PATH, "codes.json")
    code_to_str = load_codes(code_file)


    if args.split_type == 'kfold':
        codes = set([code_to_short_form(code) for code in code_to_str])
        codes.remove(0.0) # Skip "None"
        codes.remove(16.0) # Skip "Irrelevant
        codes.remove(17.0) # Skip tones
        codes.remove(18.0)
        codes.remove(19.0)
        for code in codes:
            print(code)

            train_data, test_data, test_background = get_data_split(args.split_type, p, code)
            do_all(args, train_data, test_data, test_background, code_to_str, p, code)
    else:
        train_data, test_data, test_background = get_data_split(args.split_type, p)
        do_all(args, train_data, test_data, test_background, code_to_str, p)

if __name__ == "__main__":
    main()
