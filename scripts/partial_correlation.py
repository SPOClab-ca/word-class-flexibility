"""
Script to compute partial correlations of various factors with flexibility.
"""
import numpy as np
import pandas as pd
from scipy import stats

import src.corpus
import src.partial


LANGS = "en fr zh es de ja la nl ar he id sl ro it pl da".split()
for LANG in LANGS:

  data_file = "data/wiki/processed/%s.pkl" % LANG
  corpus = src.corpus.POSCorpus.create_from_pickle(data_file_path=data_file)

  lemma_count_df = corpus.get_per_lemma_stats()
  lemma_count_df = lemma_count_df[lemma_count_df.total_count >= 100]

  lemma_count_df['log_freq'] = np.log(lemma_count_df.total_count)
  lemma_count_df['length'] = lemma_count_df.lemma.apply(lambda x: len(x))

  df = pd.get_dummies(lemma_count_df[['majority_tag', 'log_freq', 'length', 'is_flexible']], drop_first=True)

  pearson_log_freq = stats.pearsonr(df.log_freq, df.is_flexible)[0]
  pearson_length = stats.pearsonr(df.length, df.is_flexible)[0]
  pearson_maj_verb = stats.pearsonr(df.majority_tag_VERB, df.is_flexible)[0]

  partials = src.partial.calculate_partial_correlation(df)
  partial_log_freq = partials['is_flexible']['log_freq']
  partial_length = partials['is_flexible']['length']
  partial_maj_verb = partials['is_flexible']['majority_tag_VERB']

  ans = [pearson_log_freq, pearson_length, pearson_maj_verb, partial_log_freq, partial_length, partial_maj_verb]
  ans = ['%0.3f' % x for x in ans]
  print(LANG + ',' + ','.join(ans))

