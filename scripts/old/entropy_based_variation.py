"""
Calculate entropy based stats for noun and verb variation

Usage:
  python scripts/entropy_based_variation.py \
    --pkl_dir=data/wiki/processed/
    --results_dir=entropy_results/
"""
import csv
import collections
import argparse
import numpy as np
import scipy.stats
import src.corpus
import src.const


parser = argparse.ArgumentParser()
parser.add_argument('--pkl_dir', type=str)
parser.add_argument('--results_dir', type=str)
args = parser.parse_args()
print(args)

def process_language(lang):
  corpus = src.corpus.POSCorpus.create_from_pickle(data_file_path=args.pkl_dir + '/' + lang + '.pkl')
  #corpus.sentences = corpus.sentences[:len(corpus.sentences) // 50]

  # Cosine similarity between noun and verb usages
  lemma_count_df = corpus.get_per_lemma_stats()

  # Filter: must have at least [x] noun and [x] verb usages
  lemma_count_df = lemma_count_df[
    (lemma_count_df['noun_count'] >= 30) &
    (lemma_count_df['verb_count'] >= 30) &
    (lemma_count_df['is_flexible']) &
    (lemma_count_df['lemma'] != '_')
  ]
  lemma_count_df = lemma_count_df.sort_values('total_count', ascending=False)
  print('Remaining lemmas:', len(lemma_count_df))
  print('Noun lemmas:', len(lemma_count_df[lemma_count_df.majority_tag == 'NOUN']))
  print('Verb lemmas:', len(lemma_count_df[lemma_count_df.majority_tag == 'VERB']))

  def paradigm_counters(lemma):
    noun_counter = collections.Counter()
    verb_counter = collections.Counter()
    for sentence in corpus.sentences:
      for tok in sentence:
        if tok['lemma'] == lemma:
          if tok['pos'] == 'NOUN':
            noun_counter[tok['word'].lower()] += 1
          elif tok['pos'] == 'VERB':
            verb_counter[tok['word'].lower()] += 1

    noun_entropy = scipy.stats.entropy(np.array(list(noun_counter.values())))
    verb_entropy = scipy.stats.entropy(np.array(list(verb_counter.values())))
    print(lemma, noun_entropy, verb_entropy)
    return noun_entropy, verb_entropy

  lemma_count_df[['noun_entropy', 'verb_entropy']] = lemma_count_df.apply(
    lambda row: paradigm_counters(row.lemma), axis=1, result_type="expand"
  )
  return lemma_count_df


with open(args.results_dir + '/' + 'all.csv', 'w') as csvf:
  csvw = csv.writer(csvf)
  csvw.writerow(['lang', 'noun_entropy', 'verb_entropy', 'pvalue1', 'majority_entropy', 'minority_entropy', 'pvalue2'])
  for lang in src.const.LANGUAGES.keys():
    print('Processing:', lang)
    lemma_count_df = process_language(lang)
    lemma_count_df.to_csv(args.results_dir + '/' + lang + '.csv', index=False)

    noun_entropy = lemma_count_df.noun_entropy
    verb_entropy = lemma_count_df.verb_entropy
    pvalue1 = scipy.stats.ttest_rel(noun_entropy, verb_entropy)[1]

    majority_entropy = np.where(lemma_count_df.majority_tag == 'NOUN', lemma_count_df.noun_entropy, lemma_count_df.verb_entropy)
    minority_entropy = np.where(lemma_count_df.majority_tag == 'NOUN', lemma_count_df.verb_entropy, lemma_count_df.noun_entropy)
    pvalue2 = scipy.stats.ttest_rel(majority_entropy, minority_entropy)[1]

    csvw.writerow([lang, np.mean(noun_entropy), np.mean(verb_entropy), pvalue1,
      np.mean(majority_entropy), np.mean(minority_entropy), pvalue2])
