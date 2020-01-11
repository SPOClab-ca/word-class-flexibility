from collections import defaultdict, Counter
from disjoint_set import DisjointSet
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
  def create_from_ud(cls, data_file_list):
    """Initialize corpus from a path to a file in conllu format"""
    corpus = POSCorpus()
    corpus.sentences = []

    for data_file_path in data_file_list:
      with open(data_file_path, "r", encoding="utf-8") as data_file:
        data = data_file.read()
        data = conllu.parse(data)

      for token_list in data:
        sentence = []
        for token in token_list:
          pos = token['upostag']
          lemma = token['lemma']
          word = token['form']
          # Sometimes the corpus doesn't have words, only underscores
          if word == '_' or lemma == '_':
            continue
          sentence.append({'word': word, 'lemma': lemma, 'pos': pos})
        if len(sentence) > 0:
          corpus.sentences.append(sentence)

    return corpus
  

  @classmethod
  def create_from_bnc_pickled(cls, data_file_path):
    """Initialize corpus from pickled BNC corpus, generated from preprocess/process_bnc.py"""
    corpus = POSCorpus()
    with open(data_file_path, 'rb') as f:
      corpus.sentences = pickle.load(f)
    return corpus


  def get_per_lemma_stats(self, flexibility_threshold=0.05):
    # Gather usages of each lemma
    # {lemma -> (POS, word)}
    lemma_forms = defaultdict(list)
    for sentence in self.sentences:
      for token in sentence:
        lemma = token['lemma']
        word = token['word']
        pos = token['pos']
        lemma_forms[lemma].append((pos, word))

    # Noun/Verb statistics for each lemma
    lemma_count_df = []
    for lemma, lemma_occurrences in lemma_forms.items():
      noun_count = len([word for (pos, word) in lemma_occurrences if pos == 'NOUN'])
      verb_count = len([word for (pos, word) in lemma_occurrences if pos == 'VERB'])
      lemma_count_df.append({'lemma': lemma, 'noun_count': noun_count, 'verb_count': verb_count})
    lemma_count_df = pd.DataFrame(lemma_count_df)

    lemma_count_df = lemma_count_df[lemma_count_df['noun_count'] + lemma_count_df['verb_count'] > 0]
    lemma_count_df['majority_tag'] = np.where(lemma_count_df['noun_count'] >= lemma_count_df['verb_count'], 'NOUN', 'VERB')
    lemma_count_df['total_count'] = lemma_count_df[['noun_count', 'verb_count']].sum(axis=1)
    lemma_count_df['minority_count'] = lemma_count_df[['noun_count', 'verb_count']].min(axis=1)
    lemma_count_df['minority_ratio'] = lemma_count_df['minority_count'] / lemma_count_df['total_count']
    lemma_count_df['is_flexible'] = lemma_count_df['minority_ratio'] > flexibility_threshold

    return lemma_count_df


  def get_lemma_stats_merge_method(self, flexibility_threshold=0.05):

    # Helper iterator to return N/V words and lemmas in corpus, lowercased
    def iterate_words():
      for sentence in self.sentences:
        for token in sentence:
          if token['pos'] in ['NOUN', 'VERB']:
            yield token['word'].lower(), token['lemma'].lower()

    ds = DisjointSet()
    for word, lemma in iterate_words():
      ds.union(word, lemma)

    # Group words that share the same lemma
    lemma_counter = Counter()
    for _, lemma in iterate_words():
      lemma_counter[lemma] += 1

    lemma_groups = defaultdict(set)
    for word, lemma in iterate_words():
      lemma_groups[ds.find(word)].add(word)

    # Name of the group is the most frequent lemma in the group
    def get_name_for_group(word):
      maxn, maxw = 0, None
      for w in lemma_groups[ds.find(word)]:
        if lemma_counter[w] > maxn:
          maxn = lemma_counter[w]
          maxw = w
      return maxw

    # Gather usages of each lemma
    # {lemma -> (POS, word)}
    lemma_forms = defaultdict(list)
    for sentence in self.sentences:
      for token in sentence:
        lemma = token['lemma']
        word = token['word']
        pos = token['pos']
        lemma_forms[ds.find(lemma)].append((pos, word))

    # Noun/Verb statistics for each lemma
    lemma_count_df = []
    for lemma, lemma_occurrences in lemma_forms.items():
      noun_count = len([word for (pos, word) in lemma_occurrences if pos == 'NOUN'])
      verb_count = len([word for (pos, word) in lemma_occurrences if pos == 'VERB'])
      lemma_count_df.append({'lemma': get_name_for_group(lemma), 'noun_count': noun_count, 'verb_count': verb_count})
    lemma_count_df = pd.DataFrame(lemma_count_df)

    lemma_count_df = lemma_count_df[lemma_count_df['noun_count'] + lemma_count_df['verb_count'] > 0]
    lemma_count_df['majority_tag'] = np.where(lemma_count_df['noun_count'] >= lemma_count_df['verb_count'], 'NOUN', 'VERB')
    lemma_count_df['total_count'] = lemma_count_df[['noun_count', 'verb_count']].sum(axis=1)
    lemma_count_df['minority_count'] = lemma_count_df[['noun_count', 'verb_count']].min(axis=1)
    lemma_count_df['minority_ratio'] = lemma_count_df['minority_count'] / lemma_count_df['total_count']
    lemma_count_df['is_flexible'] = lemma_count_df['minority_ratio'] > flexibility_threshold

    return lemma_count_df
