"""
From ElmoPlusBert.ipynb

Usage:
  python scripts/multilingual_bert_contextual.py \
      --pkl_dir data/wiki/processed/ \
      --pkl_file en.pkl \
      --results_dir results/

python scripts/multilingual_bert_contextual.py --pkl_dir data/wiki/processed/ --pkl_file en.pkl --results_dir results/
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.decomposition
import scipy.stats
import argparse
import os

import src.corpus
import src.semantic_embedding


parser = argparse.ArgumentParser()
parser.add_argument('--pkl_dir', type=str)
parser.add_argument('--pkl_file', type=str)
parser.add_argument('--results_dir', type=str)
args = parser.parse_args()
print(args)

if not os.path.exists(args.results_dir):
  os.mkdir(args.results_dir)
outf = open(args.results_dir + '/' + args.pkl_file + '.output.txt', 'w')
print(args, file=outf)

corpus = src.corpus.POSCorpus.create_from_pickle(data_file_path=args.pkl_dir + '/' + args.pkl_file)


embedder = src.semantic_embedding.SemanticEmbedding(corpus.sentences)
embedder.init_bert(model_name='bert-base-multilingual-cased', layer=12)


# Cosine similarity between noun and verb usages
lemma_count_df = corpus.get_per_lemma_stats()

# Filter: must have at least [x] noun and [x] verb usages
lemma_count_df = lemma_count_df[(lemma_count_df['noun_count'] >= 30) & (lemma_count_df['verb_count'] >= 30)]
lemma_count_df = lemma_count_df.sort_values('total_count', ascending=False)
print('Remaining lemmas:', len(lemma_count_df), file=outf)
print('Noun lemmas:', len(lemma_count_df[lemma_count_df.majority_tag == 'NOUN']), file=outf)
print('Verb lemmas:', len(lemma_count_df[lemma_count_df.majority_tag == 'VERB']), file=outf)


lemma_count_df[['nv_cosine_similarity', 'n_variation', 'v_variation']] = lemma_count_df.apply(lambda row: embedder.get_contextual_nv_similarity(row.lemma, method="bert"),
                       axis=1, result_type="expand")
lemma_count_df.to_csv(args.results_dir + '/' + args.pkl_file + '.lemma_count_df.csv', index=False)


# Difference in similarity when base is noun vs verb
plt.clf()
plot = sns.distplot(lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].nv_cosine_similarity, label='Base=N')
plot = sns.distplot(lemma_count_df[lemma_count_df.majority_tag == 'VERB'].nv_cosine_similarity, label='Base=V')
plt.legend()
plot.set(title="Average Cosine Similarity between Noun/Verb Usage",
         xlabel="Cosine Similarity", ylabel="Count")
plt.savefig(args.results_dir + '/' + args.pkl_file + '.semantic-shift.png')

print('Mean cosine similarity when Base=N:', np.mean(lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].nv_cosine_similarity), file=outf)
print('Mean cosine similarity when Base=V:', np.mean(lemma_count_df[lemma_count_df.majority_tag == 'VERB'].nv_cosine_similarity), file=outf)


# T-test of difference in mean
print(scipy.stats.ttest_ind(lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].nv_cosine_similarity,
                            lemma_count_df[lemma_count_df.majority_tag == 'VERB'].nv_cosine_similarity), file=outf)


# Difference in variation between majority and minority class
majority_variation = np.where(lemma_count_df.majority_tag == 'NOUN', lemma_count_df.n_variation, lemma_count_df.v_variation)
minority_variation = np.where(lemma_count_df.majority_tag == 'NOUN', lemma_count_df.v_variation, lemma_count_df.n_variation)
plt.clf()
plot = sns.distplot(majority_variation, label='Majority')
plot = sns.distplot(minority_variation, label='Minority')
plt.legend()
plot.set(title="Semantic variation within majority and minority POS class",
         xlabel="Standard deviation", ylabel="Density")
plt.savefig(args.results_dir + '/' + args.pkl_file + '.majority_minority_variation.png')


print('Mean majority variation:', np.mean(majority_variation), file=outf)
print('Mean minority variation:', np.mean(minority_variation), file=outf)

# Paired t-test for difference
print(scipy.stats.ttest_rel(majority_variation, minority_variation), file=outf)
outf.close()
