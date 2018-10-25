



FINAL_PARAMS = [
50000, 500, 50, 250, 500, 0.4, 3
]

RUSSIAN_PARAMS = [
50000, 500, 50, 250, 1000, 0.3, 3
]

class Params:
   def  __init__(self, p = FINAL_PARAMS):
       self.VOCAB_SIZE = p[0]
       self.MIN_CUT = p[1]
       self.MAX_CUT = p[2]
       self.TO_RETURN_COUNT = p[3]
       self.VEC_SEARCH = p[4]
       self.SIM_THRESH = p[5]
       self.LEX_COUNT = p[6]

       # set various path names
       # This should contain the annotated versions of
       # immigration.json, tobacco.json, samesex.json, and codes.json
       self.MFC_PATH = "/usr1/home/anjalief/corpora/media_frames_corpus/"

       # This is the path to raw MFC data (not limited to annotated subset)
       # This data is used for obtaining counts and training word embeddings
       # Within the "parsed" directory, there should be subfolers: samesex, smoking, immigation
       # Within the sub-directories there should be a folder called "json" that has scraped json files

       # Ex /usr1/home/anjalief/media_frames_corpus/parsed/samesex/json/*.json
       self.MFC_RAW_PATH = "/usr1/home/anjalief/media_frames_corpus/parsed/"


       # Corpus used to train English word embedding model
       self.NYT_PATH = "/usr1/home/anjalief/corpora/nyt_news/all_nyt_paras_text/*/*.txt.tok"


       # Name our cached files; these names shouldn't much matter
       self.ENGLISH_BASE_MODEL = "./cache/nyt_base.model" # Word embedding model trained on a background corpus, generated with make_nyt_embed.py
       self.ENGLISH_MFC_MODEL = "./cache/nyt_mfc.model"



