#!/usr/bin/env python3

import os
import glob
import math
from scipy.stats.stats import pearsonr
from scipy import spatial
from datetime import date
from collections import defaultdict

NEW_ARTICLE_TOKEN="NEW - ARTICLE - TOKEN"

def ArticleIter(filename, new_article_token, verbose=False):
  current_article = []
  if verbose:
    print("Processing", filename)
  for line in open(filename):
    line = line.strip()
    if not line:
      continue
    if line == new_article_token:
      if current_article:
        yield "\n".join(current_article)
        current_article = []
    else:
      current_article.append(line)
  if current_article:
    yield "\n".join(current_article)

def LoadArticles(article_glob, verbose=True, new_article_token=NEW_ARTICLE_TOKEN, sort_files=False, split = False):
  articles = []
  article_index = []
  if sort_files:
    files = sorted(glob.iglob(article_glob))
  else:
    files = glob.iglob(article_glob)
  for filename in files:
    if verbose:
      print("Loading:", filename)
    for index, article in enumerate(ArticleIter(filename, new_article_token, verbose)):
      if split:
        articles.append(article.split())
      else:
        articles.append(article)
      article_index.append((filename, index))
    if verbose:
      print("  articles:", index+1)
  return articles, article_index

def Similarity(v1, v2, metric="cosine"):
  def IsZero(v):
    return all(n == 0 for n in v)

  if metric == "correlation":
    if IsZero(v1) or IsZero(v2):
      return 0.0
    return pearsonr(v1, v2)[0]

  if metric == "abs_correlation":
    if IsZero(v1) or IsZero(v2):
      return 0.0
    return abs(pearsonr(v1, v2)[0])

  if metric == "cosine":
    return spatial.distance.cosine(v1, v2)

def LoadVectors(filename):
  vectors = []
  for line in open(filename):
    vector = [float(n) for n in line.split()]
    if len(vector) == 0:
      continue
    # normalize
    sqrt_length = math.sqrt(sum([n**2 for n in vector]) + 1e-6)
    vectors.append([n/sqrt_length for n in vector])
  return vectors

def GetSimilarArticles(articles, vectors, gold_vector, threshold):
  similar = []
  for article, vector in zip(articles, vectors):
    if Similarity(vector, gold_vector) < threshold:
      similar.append(article)
  return similar


# Note, this is intended for input into gensim models, can't make it
# a full generator because gensim wants an iterator
import nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#from russian_stemmer import country_russian_stemmer
class SentenceIter(object):
  def __init__(self, article_glob, verbose=True, new_article_token=NEW_ARTICLE_TOKEN, skip_corrections=False):
    self.article_glob = article_glob
    self.verbose = verbose
    self.new_article_token = new_article_token
    self.skip_corrections = skip_corrections
#    self.stemmer = country_russian_stemmer()

  def __iter__(self):
    for filename in glob.iglob(self.article_glob):
      if self.skip_corrections and "Corrections" in filename:
        continue
      if self.verbose:
        print("Loading:", filename)
      for article in ArticleIter(filename, self.new_article_token, self.verbose):
        for s in tokenizer.tokenize(article):
          # NOTE: We almost certainly want to use the ispras lemmatizer here,
          # but it's going to be very slow
          yield s.split()

# for ex path/2007_1.txt.tok, return 2007, 1
def get_year_month(file_path):
  filename = os.path.basename(file_path)
  year_month = filename.split('.')[0]
  splits = year_month.split('_')
  return int(splits[0]), int(splits[1])

# Assigns all dates to the first of the month
def get_date(file_path):
  year, month = get_year_month(file_path)
  return date(year, month, 1)

# Return year and Q1, Q2, Q3, Q4
def get_quarter(file_path):
  year, month = get_year_month(file_path)
  return year, ((month - 1) / 3) + 1

# return a date sequence and a list of lists, that groups files
# in the same month together (ex. Izvestiia and Pravda 1/2004)
def get_monthly_filenames(input_path, suffix=".txt.tok"):
    date_seq = []
    filenames = [] # list of lists, grouped by date
    date_to_filename = {}

    file_glob = input_path + "*_*" + suffix
    for filename in glob.iglob(file_glob):
      year, month = get_year_month(filename)
      date_to_filename[date(int(year), int(month), 1)] = filename

    for key in sorted(date_to_filename):
      date_seq.append(key)
      filenames.append([date_to_filename[key]])

    return date_seq, filenames

def quarter_to_month(q):
  # 1 2 3
  if q == 1:
    return 1
  # 4 5 6
  if q == 2:
    return 4
  # 7 8 9
  if q == 3:
    return 7
  if q == 4:
    return 10

# return a date sequence and a list of lists, that groups files
# in the same quarter together (ex. Jan, Feb, March 2003; Izvestiia and Pravda)
def get_quarterly_filenames(input_path, suffix=".txt.tok"):
    quarter_to_filenames = defaultdict(list)

    file_glob = input_path + "*_*" + suffix
    for filename in glob.iglob(file_glob):
      y, quarter = get_quarter(filename)
      key = date(int(y), quarter_to_month(int(quarter)), 1)
      quarter_to_filenames[key].append(filename)

    date_seq = []
    filenames = []
    for key in sorted(quarter_to_filenames):
        date_seq.append(key)
        filenames.append(quarter_to_filenames[key])
    return date_seq, filenames

# return a date sequence and a list of lists, that groups files
# in the same quarter together (ex. Jan, Feb, March 2003; Izvestiia and Pravda)
def get_semi_filenames(input_path, suffix=".txt.tok"):
    semi_to_filenames = defaultdict(list)

    file_glob = input_path + "*_*" + suffix
    for filename in glob.iglob(file_glob):
      y,month = get_year_month(filename)
      if int(month) < 7:
        semi = 1
      else:
        semi = 7
      key = date(int(y), semi, 1)
      semi_to_filenames[key].append(filename)

    date_seq = []
    filenames = []
    for key in sorted(semi_to_filenames):
        date_seq.append(key)
        filenames.append(semi_to_filenames[key])
    return date_seq, filenames

def get_yearly_filenames(input_path, suffix=".txt.tok"):
    year_to_filenames = defaultdict(list)

    # I fear I am breaking all my code with this
    file_glob = input_path + "*_*" + suffix
    for filename in glob.iglob(file_glob):
      year = os.path.basename(filename).split(".")[0].split("_")[0]
      key = date(int(year), 1, 1)
      year_to_filenames[key].append(filename)

    date_seq = []
    filenames = []
    for key in sorted(year_to_filenames):
        date_seq.append(key)
        filenames.append(year_to_filenames[key])
    return date_seq, filenames

def get_ordered_files(input_path):
  article_glob = os.path.join(input_path + "*")
  date_to_name = {}
  for filename in glob.iglob(article_glob):
    try:
      year, month = get_year_month(filename)
      date_to_name[date(year, month, 1)] = filename
    # we might have other junk in the folder (ex. base_model)
    except:
      continue

  date_seq = []
  filenames = []
  for key in sorted(date_to_name):
    date_seq.append(key)
    filenames.append(date_to_name[key])
  return date_seq, filenames

def get_files_by_time_slice(input_path, timestep, suffix=".txt.tok"):
    if timestep == "monthly":
        date_seq, filenames = get_monthly_filenames(input_path, suffix)
    elif timestep == "quarterly":
        date_seq, filenames = get_quarterly_filenames(input_path, suffix)
    elif timestep == "semi":
        date_seq, filenames = get_semi_filenames(input_path, suffix)
    else:
        assert (timestep == "yearly")
        date_seq, filenames = get_yearly_filenames(input_path, suffix)
    assert (len(date_seq) == len(filenames))
    return date_seq, filenames
