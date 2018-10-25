import json

from nltk import tokenize, data
from collections import defaultdict, Counter

from sklearn.model_selection import StratifiedKFold

import random
sentence_tokenizer = data.load('tokenizers/punkt/english.pickle')

random.seed(0)

def code_to_short_form(frame):
  if frame == "null" or frame is None:
    return frame
  f = str(frame).split(".")
  return float(f[0] + ".0")

def load_codes(filename):
    str_to_code = json.load(open(filename))
    return {float(k) : str_to_code[k] for k in str_to_code}

def load_json_as_list(input_files):
    all_text = []
    for filename in input_files:
        json_text = json.load(open(filename))
        # no point in keeping files that aren't framing annotated
        all_text += [x for _,x in json_text.items() if x["annotations"].get("framing", {}) != {}]
    return all_text

# For each doc, return text and all frames annotated in doc (by
# any annotator), and primary frame
# can give filenames or raw json as input
class FrameAnnotationsIter(object):
  def __init__(self, json_text, verbose=False):
      self.verbose = verbose
      self.json_text = json_text

  def __iter__(self):
      for annotated_file in self.json_text:
          assert "framing" in annotated_file["annotations"] and annotated_file["annotations"]["framing"] != {}
          text = annotated_file["text"].lower()

          # we return all frames anybody found in the document
          frames = set()
          for annotation_set in annotated_file["annotations"]["framing"]:
              for frame in annotated_file["annotations"]["framing"][annotation_set]:
                  frames.add(code_to_short_form(frame["code"]))
          yield tokenize.word_tokenize(text), frames, code_to_short_form(annotated_file["primary_frame"])

# For each doc, return text and frames that all annotators marked and
# frames that any annotator marked
# Used to measure doc-level hard and soft recall
class FrameHardSoftIter(object):
  def __init__(self, json_text, all_frames, verbose=False):
      self.verbose = verbose
      self.json_text = json_text
      self.all_frames = all_frames

  def __iter__(self):
      for annotated_file in self.json_text:
          assert "framing" in annotated_file["annotations"] and annotated_file["annotations"]["framing"] != {}
          text = annotated_file["text"].lower()

          # we return all frames anybody found in the document
          frame_count = Counter()
          for annotation_set in annotated_file["annotations"]["framing"]:
              for frame in annotated_file["annotations"]["framing"][annotation_set]:
                  frame_count[code_to_short_form(frame["code"])] += 1
          frame_to_all = {}
          frame_to_any = {}
          for frame in self.all_frames:
              frame_to_all[frame] = False
              frame_to_any[frame] = False
              # this means all annotators marked it as present
              if frame_count[frame] == len(annotated_file["annotations"]["framing"]):
                  frame_to_all[frame] = True
              if frame_count[frame] > 0:
                  frame_to_any[frame] = True

          yield tokenize.word_tokenize(text), frame_to_all, frame_to_any

class BackgroundIter(object):
  def __init__(self, input_files, verbose=False, sent_level=True):
      self.input_files = input_files
      self.verbose = verbose
      self.sent_level = sent_level

  def __iter__(self):
      for filename in self.input_files:
          if self.verbose:
              print("Loading:", filename)

          json_text = json.load(open(filename))
          if not self.sent_level:
              all_words = []
              for sentence in json_text["BODY"]:
                  all_words += tokenize.word_tokenize(sentence.lower())
              yield all_words
          else:
              for sentence in json_text["BODY"]:
                  yield tokenize.word_tokenize(sentence.lower())

# This is super confusing but I think what we want is take the text
# that uses a frame and take all text that doesn't use a frame
# all text that uses frame is easy
# all text that doesn't use a frame, we can either take annotated spans,
# or we can take sentences. Let's take sentences
def get_sentence_level_test(json_text, all_frames):
    frame_to_contains = defaultdict(list)
    frame_to_doesnt = defaultdict(list)
    for annotated_file in json_text:
        assert "framing" in annotated_file["annotations"] and annotated_file["annotations"]["framing"] != {}
        text = annotated_file["text"].lower()

        # the tokenizer cuts some whitespace. We're just going to go with . divisions
        # I think that's sufficient for testing
        # sentences = sentence_tokenizer.tokenize(text)
        sentences = [s + "." for s in text.replace("?",".").replace("!",".").split('.')]
        # last sentence might not have a period. Cut off the one we added
        if sentences[-1][-1] != text[-1]:
            sentences[-1] = sentences[-1][:-1]
        q = sum([len(s) for s in sentences])
        assert(q == len(text)), str(q) + " " + str(len(text)) + " " + str(len(sentences))

        start_idx = 0
        for s in sentences:
            end_idx = start_idx + len(s)

            frames_in_sentence = set()

            for annotation_set in annotated_file["annotations"]["framing"]:
                for frame in annotated_file["annotations"]["framing"][annotation_set]:
                    code = code_to_short_form(frame["code"])

                    # easy part we KNOW this text uses this frame
                    # only add it once
                    frame_start = int(frame["start"])
                    frame_end = int(frame["end"])
                    if start_idx == 0:
                        coded_text = text[frame_start:frame_end]
                        frame_to_contains[code].append(tokenize.word_tokenize(coded_text))

                    # if either the start or end are inside the text, then we have overlap
                    if (frame_start >= start_idx and frame_start < end_idx) or \
                          (frame_end >= start_idx and frame_end < end_idx):
                        frames_in_sentence.add(code)

            for f in all_frames:
                if not f in frames_in_sentence:
                    frame_to_doesnt[f].append(tokenize.word_tokenize(s))

            start_idx = end_idx # move to next sentence
        assert end_idx == len(text), str(end_idx) + " " + str(len(text))
    return frame_to_contains, frame_to_doesnt


# This is for the random splitter. We're going to load all the code and
# split into train and test sets
# First just do it once randomly
# returns test_set, training set
def get_random_split(input_files, fold = 0, num_folds = 5, filter_tone = False):
    random.seed(0)
    # first load all data
    all_text = load_json_as_list(input_files)
    if filter_tone:
      all_text = [t for t in all_text if t["primary_frame"] is not None and t["primary_tone"] is not None]

    # split is 80/20
    random.shuffle(all_text)

    num_test = int(len(all_text) / num_folds)
    test_split_start = fold * num_test
    test_split_end = test_split_start + num_test

    return all_text[test_split_start:test_split_end], all_text[:test_split_start] + all_text[test_split_end:]

# Then do stratified K-Fold for each frame seperately
def get_per_frame_split(input_files, frame_short_form, fold_num=0):
    assert (fold_num < 5)
    splitter = StratifiedKFold(n_splits=5)
    all_text = load_json_as_list(input_files)
    random.shuffle(all_text)

    dummy_text = []
    labels = []

    frame_iter = FrameAnnotationsIter(all_text)
    for text, frames, _ in frame_iter:
        dummy_text.append("dummy")
        labels.append(frame_short_form in frames)

    # for now we just grab the first fold
    which_fold = 0
    for train_indices, test_indices in splitter.split(dummy_text, labels):
        while which_fold < fold_num:
            which_fold += 1
            continue
        assert (which_fold == fold_num)
        train_data = [all_text[i] for i in train_indices]
        test_data = [all_text[i] for i in test_indices]
        break
    assert (len(test_data) + len(train_data)) == len(all_text)
    return test_data, train_data

def main():
  files = ["/usr1/home/anjalief/corpora/media_frames_corpus/samesex.json", "/usr1/home/anjalief/corpora/media_frames_corpus/tobacco.json", "/usr1/home/anjalief/corpora/media_frames_corpus/immigration.json"]
#  get_random_split(files)
  test_data, train_data = get_random_split(files)
  count_in = 0
  count_total = 0
  train_count_in = 0
  train_count_total = 0
  for text, frames, _ in FrameAnnotationsIter(test_data):
      count_total += 1

  for text, frames, _ in FrameAnnotationsIter(train_data):
      train_count_total += 1

  print(train_count_total + count_total)

if __name__ == "__main__":
  main()
