from datetime import date
import argparse
from collections import defaultdict, Counter
import os
import sys
sys.path.append("..")
from article_utils import *
from get_dates import *
from parse_frames import get_article_top_words
import operator

from params import Params
from utils import is_adjective,is_noun

BIGRAMS=False
LIMIT=False

import pickle

params = Params()

# Build counters for external articles in provided filenames
def load_counts_by_frame(filenames, frame_to_lex, primary_only=False):
  frame_to_counter = defaultdict(Counter)

  for filename in filenames:
      articles,_ = LoadArticles(filename, verbose=False)
      for article in articles:
          words = article.split()
          word_counter = Counter(words)

          # Only look at USA articles
          if word_counter["USA"] < 2:
              continue

          if primary_only:
              sums = []
              for f in frame_to_lex:
                  sums.append((f, sum([word_counter[w] for w in frame_to_lex[f]])))
              frame, word_count = max(sums, key=operator.itemgetter(1))
              frame_to_counter[frame].update(word_counter)
          else:
              for frame in frame_to_lex:
                  # this article is relevant to this frame. update the counter
                  if sum([word_counter[x] for x in frame_to_lex[frame]]) > params.LEX_COUNT:
                      frame_to_counter[frame].update(word_counter)

  return frame_to_counter

def load_internal_external_counts_by_frame(filenames, frame_to_lex, primary_only=False):
    frame_to_internal_counter = defaultdict(Counter)
    frame_to_external_counter = defaultdict(Counter)

    for filename_group in filenames:
      for filename in filename_group:
        articles,_ = LoadArticles(filename, verbose=False)
        for article in articles:
            words = article.split()
            word_counter = Counter(words)

            if primary_only:
              sums = []
              for f in frame_to_lex:
                  sums.append((f, sum([word_counter[w] for w in frame_to_lex[f]])))
              frame, word_count = max(sums, key=operator.itemgetter(1))
              if word_counter["USA"] >= 2:
                  frame_to_external_counter[frame].update(word_counter)
              else:
                  frame_to_internal_counter[frame].update(word_counter)
            else:
              for frame in frame_to_lex:
                  # this article is relevant to this frame. update the counter
                  if sum([word_counter[x] for x in frame_to_lex[frame]]) > params.LEX_COUNT:
                      if word_counter["USA"] >= 2:
                          frame_to_external_counter[frame].update(word_counter)
                      else:
                          frame_to_internal_counter[frame].update(word_counter)

    return frame_to_internal_counter, frame_to_external_counter


# We use all filenames
def LoadBackgroundCorpus(input_path, timestep = "monthly", cache_file_name = "background_cache.pickle"):

  if os.path.isfile(cache_file_name):
    return pickle.load(open(cache_file_name, "rb"))

  _,all_files = get_files_by_time_slice(input_path, timestep)

  word_to_count = Counter()
  for filenames in all_files:
    for filename in filenames:
      articles, _ = LoadArticles(filename)
      for article in articles:
        words = article.split()
        word_to_count.update(make_counter(words))

  pickle.dump(word_to_count, open(cache_file_name, "wb"))
  return word_to_count

def write_log_odds(counts1, counts2, prior, outfile_name = None):
    # COPIED FROM LOG_ODDS FILE
    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1  = sum(counts1.values())
    n2  = sum(counts2.values())
    nprior = sum(prior.values())


    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    if outfile_name:
      outfile = open(outfile_name, 'w')
      for word in sorted(delta, key=delta.get):
        outfile.write(word)
        outfile.write(" %.3f\n" % delta[word])

      outfile.close()
    else:
      return delta


def dates_to_files(dates, input_path):
  return [os.path.join(input_path,
                       str(d.year) + "_" + str(d.month) + ".txt.tok")
          for d in dates]

# Take log odds of articles in USA vs. non-USA
def other_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent_change", default="/usr1/home/anjalief/corpora/russian/percent_change/russian_rtsi_rub.csv")
    parser.add_argument("--input_path", default="/usr1/home/anjalief/corpora/russian/yearly_mod_subs/iz_lower/init_files/")
    parser.add_argument("--framing_lex", default="./cache/russian_params.pickle")
    args = parser.parse_args()

    frame_to_lex = pickle.load(open(args.framing_lex, "rb"))
    top_words, num_articles, article_counter = get_article_top_words(args.input_path + "*.txt.tok")
    vocab = sorted(top_words, key=top_words.get, reverse = True)[:params.VOCAB_SIZE]

    _,all_files = get_files_by_time_slice(args.input_path, "monthly")
    frame_to_internal_counter, frame_to_external_counter = load_internal_external_counts_by_frame(all_files, frame_to_lex)

    prior = LoadBackgroundCorpus(args.input_path)

    for frame in frame_to_lex:
        delta = write_log_odds(frame_to_internal_counter[frame], frame_to_external_counter[frame], prior)

        internal_1000 = sorted(delta, key=delta.get)[:200]
        external_1000 = sorted(delta, key=delta.get)[-200:]

        print(frame)
        print("INTERNAL", internal_1000)
        print("EXTERNAL", external_1000)
        print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent_change", default="/usr1/home/anjalief/corpora/russian/percent_change/russian_rtsi_rub.csv")
    parser.add_argument("--input_path", default="/usr1/home/anjalief/corpora/russian/yearly_mod_subs/iz_lower/init_files/")
    parser.add_argument("--framing_lex", default="./cache/russian_params.pickle")
    args = parser.parse_args()


    # Intersect with framing lexicons
    prior = LoadBackgroundCorpus(args.input_path)

    good_dates, bad_dates = get_month_prev(args.percent_change)
    upturn_good, upturn_bad = get_good_month_prev(args.percent_change, percent=6)

    good_file_names = dates_to_files(good_dates, args.input_path)
    bad_file_names = dates_to_files(bad_dates, args.input_path)

    upturn_good_file_names = dates_to_files(upturn_good, args.input_path)
    upturn_bad_file_names = dates_to_files(upturn_bad, args.input_path)

    frame_to_lex = pickle.load(open(args.framing_lex, "rb"))
    top_words, num_articles, article_counter = get_article_top_words(args.input_path + "*.txt.tok")
    vocab = sorted(top_words, key=top_words.get, reverse = True)[:params.VOCAB_SIZE]
#    vocab = [v for v in vocab if is_noun(v)]

    frame_to_good_counts = load_counts_by_frame(good_file_names, frame_to_lex)
    frame_to_bad_counts = load_counts_by_frame(bad_file_names, frame_to_lex)

    frame_to_upturn_good = load_counts_by_frame(upturn_good_file_names, frame_to_lex)
    frame_to_upturn_bad = load_counts_by_frame(upturn_bad_file_names, frame_to_lex)

    for frame in frame_to_lex:
        delta = write_log_odds(frame_to_good_counts[frame], frame_to_bad_counts[frame], prior)
        delta_upturn = write_log_odds(frame_to_upturn_good[frame], frame_to_upturn_bad[frame], prior)

        # limit to common words
        delta = {v:delta[v] for v in vocab}

        # Smallest words are most negative; therefore they are more common in bad months
        # Positive words are most closely associated with good
        # bad_1000 = sorted(delta, key=delta.get)[:500]
        # good_1000 = sorted(delta, key=delta.get)[-500:]
        bad_1000 = [d for d in delta if delta[d] < -0.49]
        good_1000 = [d for d in delta if delta[d] > 0.49]


        # Now overlap with topics that become less common after upturns
        delta_upturn = {v:delta_upturn[v] for v in vocab}

        # bad_upturn_1000 = sorted(delta_upturn, key=delta_upturn.get)[:500]
        # good_upturn_1000 = sorted(delta_upturn, key=delta_upturn.get)[-500:]
        bad_upturn_1000 = [d for d in delta if delta_upturn[d] < -0.49]
        good_upturn_1000 = [d for d in delta if delta_upturn[d] > 0.49]

        # Final lex: words that become more common after downturns and less common after upturns
        # (good lex is reversed)
        final_good = [word for word in good_1000 if word in good_upturn_1000]
        final_bad = [word for word in bad_1000 if word in bad_upturn_1000]
        print(frame, len(final_good), len(final_bad))
        print("GOOD", final_good, sum([delta[w] for w in final_good])/len(final_good),  sum([delta_upturn[w] for w in final_good])/len(final_good))
        print("BAD", final_bad,  sum([delta[w] for w in final_bad])/len(final_bad),  sum([delta_upturn[w] for w in final_bad])/len(final_bad))
        print()

    return

if __name__ == "__main__":
   main()
