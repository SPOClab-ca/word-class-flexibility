from collections import defaultdict
import numpy as np
import pandas as pd
import conllu
import pickle


class POSCorpus(object):
  """Corpus for analyzing POS flexibility. After creation, corpus.sentences should consist of
  a list of sentences, each with a list of words. Example structure:
  [
    [
      {'word': "I", 'lemma': "i", 'pos': "PRON"},
      {'word': "love", 'lemma': "love", 'pos': "VERB"},
      {'word': "cats", 'lemma': "cat", 'pos': "NOUN"},
    ]
  ]
  """
  def __init__(self):
    pass

  @classmethod
  def create_from_ud(cls, data_file_path):
    """Initialize corpus from a path to a file in conllu format"""
    corpus = POSCorpus()
    with open(data_file_path, "r", encoding="utf-8") as data_file:
      corpus.data = data_file.read()
      corpus.data = conllu.parse(corpus.data)

    corpus.sentences = []
    for token_list in corpus.data:
      sentence = []
      for token in token_list:
        pos = token['upostag']
        lemma = token['lemma']
        word = token['form']
        sentence.append({'word': word, 'lemma': lemma, 'pos': pos})
      corpus.sentences.append(sentence)

    return corpus
  

  @classmethod
  def create_from_bnc_pickled(cls, data_file_path):
    """Initialize corpus from pickled BNC corpus, generated from preprocess/process_bnc.py"""
    corpus = POSCorpus()
    with open(data_file_path, 'rb') as f:
      corpus.sentences = pickle.load(f)
    return corpus


  def get_per_lemma_stats(self, POS1='NOUN', POS2='VERB', flexibility_threshold=0.05):
    # Gather usages of each lemma
    # {lemma -> (POS, word)}
    lemma_forms = defaultdict(list)
    for sentence in self.sentences:
      for token in sentence:
        lemma = token['lemma']
        word = token['word']
        pos = token['pos']
        lemma_forms[lemma].append((pos, word))

    # Part-of-speech statistics for each lemma
    lemma_count_df = []
    for lemma, lemma_occurrences in lemma_forms.items():
      pos1_count = len([word for (pos, word) in lemma_occurrences if pos == POS1])
      pos2_count = len([word for (pos, word) in lemma_occurrences if pos == POS2])
      lemma_count_df.append({'lemma': lemma, 'pos1_count': pos1_count, 'pos2_count': pos2_count})
    lemma_count_df = pd.DataFrame(lemma_count_df)

    lemma_count_df = lemma_count_df[lemma_count_df['pos1_count'] + lemma_count_df['pos2_count'] > 0]
    lemma_count_df['majority_tag'] = np.where(lemma_count_df['pos1_count'] >= lemma_count_df['pos2_count'], POS1, POS2)
    lemma_count_df['total_count'] = lemma_count_df[['pos1_count', 'pos2_count']].sum(axis=1)
    lemma_count_df['minority_count'] = lemma_count_df[['pos1_count', 'pos2_count']].min(axis=1)
    lemma_count_df['minority_ratio'] = lemma_count_df['minority_count'] / lemma_count_df['total_count']
    lemma_count_df['is_flexible'] = lemma_count_df['minority_ratio'] > flexibility_threshold

    return lemma_count_df
