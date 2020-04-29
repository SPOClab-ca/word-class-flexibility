"""
Usage:
  python scripts/entropy_based_variation.py \
    --pkl_dir=data/wiki/processed/
"""
import collections
import argparse
import src.corpus


parser = argparse.ArgumentParser()
parser.add_argument('--pkl_dir', type=str)
args = parser.parse_args()
print(args)

corpus = src.corpus.POSCorpus.create_from_pickle(data_file_path=args.pkl_dir + '/' + 'en.pkl')

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
  return noun_counter, verb_counter

noun_counts, verb_counts = paradigm_counters('work')
print(noun_counts)
print(verb_counts)
