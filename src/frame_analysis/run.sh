# FRAMING
# This script broadly requires that the hard-coded paths in params.py be set correctly
#######################################################################################################################

################################################# ENGLISH EVALUATIONS #################################################
# We use word embedding models for query expansion. First, generate a generic model trained on a background
# corpus (we use NYT data). We later update this model with MFC data
python make_nyt_embed.py

# Runs evaluations described in paper
python cross_eval_frames.py --split_type dcard

# Logistic regression baseline from Table 6
python baseline.py


##################################################### RUSSIAN ANALYSIS ###############################################

# Generate Figure 2
python russian_pmi.py --whole_corpus --frame_lex ./cache/russian_params.pickle