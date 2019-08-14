from collections import defaultdict
import numpy as np
import pandas as pd
import conllu


class UDCorpus(object):
  def __init__(self, data_file_path):
    with open(data_file_path, "r", encoding="utf-8") as data_file:
      self.data = data_file.read()
      self.data = conllu.parse(self.data)

  def get_per_lemma_stats(self, flexibility_threshold=0.05):
    # Gather usages of each lemma
    # {lemma -> (POS, word, sentence)}
    lemma_forms = defaultdict(list)
    for token_list in self.data:
      sentence = ' '.join([t['form'] for t in token_list])
      for token in token_list:
        pos_tag = token['upostag']
        lemma = token['lemma']
        word = token['form']
        lemma_forms[lemma].append((pos_tag, word, sentence))

    # Noun/Verb statistics for each lemma
    lemma_count_df = []
    for lemma, lemma_occurrences in lemma_forms.items():
      noun_count = len([word for (pos, word, _) in lemma_occurrences if pos == 'NOUN'])
      verb_count = len([word for (pos, word, _) in lemma_occurrences if pos == 'VERB'])
      lemma_count_df.append({'lemma': lemma, 'noun_count': noun_count, 'verb_count': verb_count})
    lemma_count_df = pd.DataFrame(lemma_count_df)

    lemma_count_df = lemma_count_df[lemma_count_df['noun_count'] + lemma_count_df['verb_count'] > 0]
    lemma_count_df['majority_tag'] = np.where(lemma_count_df['noun_count'] >= lemma_count_df['verb_count'], 'NOUN', 'VERB')
    lemma_count_df['total_count'] = lemma_count_df[['noun_count', 'verb_count']].sum(axis=1)
    lemma_count_df['minority_count'] = lemma_count_df[['noun_count', 'verb_count']].min(axis=1)
    lemma_count_df['minority_ratio'] = lemma_count_df['minority_count'] / lemma_count_df['total_count']
    lemma_count_df['is_flexible'] = lemma_count_df['minority_ratio'] > flexibility_threshold

    return lemma_count_df
