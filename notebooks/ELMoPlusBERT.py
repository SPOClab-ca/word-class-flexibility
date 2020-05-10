#!/usr/bin/env python
# coding: utf-8

# # ELMo and BERT Contextual Embeddings
# 
# In this notebook, we use contextual embeddings from ELMo/BERT to study semantic change of conversion.

# In[1]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.decomposition
import random
import scipy.stats

import src.corpus
import src.semantic_embedding

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Parse the corpus

# In[2]:


#UD_PATH = '../data/ud_all/ud-treebanks-v2.5/'
#ud_files = src.corpus.group_treebanks_by_language(UD_PATH)
#corpus = src.corpus.POSCorpus.create_from_ud(data_file_list=ud_files['English'])

BNC_FILE = "../data/wiki/processed/en.pkl"
#BNC_FILE = "../data/bnc/bnc.pkl"
corpus = src.corpus.POSCorpus.create_from_pickle(data_file_path=BNC_FILE)


# ## Compute embeddings on random part of the corpus

# In[3]:


# Take only 1M words out of 4M to make it run faster
SAMPLE_PROPORTION = 1.0
random.seed(12345)
random_indices = random.sample(range(len(corpus.sentences)), int(SAMPLE_PROPORTION * len(corpus.sentences)))

sampled_sentences = []
for ix in random_indices:
  sampled_sentences.append(corpus.sentences[ix])
  
embedder = src.semantic_embedding.SemanticEmbedding(sampled_sentences)
embedder.init_bert(model_name='bert-base-uncased', layer=12)
#embedder.init_elmo()


# ## Compute embeddings of instances of a fixed lemma

# In[4]:


FIXED_LEMMA = "talk"
noun_embeddings, verb_embeddings, noun_indices, verb_indices = embedder.get_bert_embeddings_for_lemma(FIXED_LEMMA)
print("Noun instances:", noun_embeddings.shape[0])
print("Verb instances:", verb_embeddings.shape[0])


# ## Apply PCA and plot

# In[5]:


pca = sklearn.decomposition.PCA(n_components=2)
all_embeddings = pca.fit_transform(np.vstack([noun_embeddings, verb_embeddings]))
all_embeddings_df = pd.DataFrame({'x0': all_embeddings[:,0], 'x1': all_embeddings[:,1]})
all_embeddings_df['pos'] = ['Noun'] * len(noun_embeddings) + ['Verb'] * len(verb_embeddings)
all_embeddings_df['sentence_ix'] = noun_indices + verb_indices


# In[ ]:


plot = sns.scatterplot(data=all_embeddings_df.sample(min(len(all_embeddings), 1000)),
                       x='x0', y='x1', hue='pos', linewidth=0, s=20,
                       palette=sns.color_palette("muted", n_colors=2))
handles, labels = plot.get_legend_handles_labels()
plot.legend(handles=handles[1:], labels=labels[1:], loc="upper right")
plt.suptitle("Lemma: %s" % FIXED_LEMMA, fontsize=14, y=0.95)
plt.xlabel('PC1')
plt.ylabel('PC2')
plot.axes.get_xaxis().set_ticks([])
plot.axes.get_yaxis().set_ticks([])
#plt.savefig('bert-work.pdf')
plt.show()


# ## Utility to inspect what it's capturing
num_printed = 0
for _, row in all_embeddings_df.iterrows():
  if row.x0 > 14: # <- Put whatever condition here
    sent = sampled_sentences[row.sentence_ix]
    print("POS=" + row.pos + ";", ' '.join([t['word'] for t in sent]))
    num_printed += 1
  if num_printed > 10:
    break
# ## Cosine similarity between noun and verb usages

# In[7]:


lemma_count_df = corpus.get_per_lemma_stats()

# Filter: must have at least [x] noun and [x] verb usages
lemma_count_df = lemma_count_df[(lemma_count_df['noun_count'] >= 30) & (lemma_count_df['verb_count'] >= 30)]
lemma_count_df = lemma_count_df.sort_values('total_count', ascending=False)
#lemma_count_df = lemma_count_df[~lemma_count_df.lemma.isin(['go', 'will', 'may'])]
print('Remaining lemmas:', len(lemma_count_df))
print('Noun lemmas:', len(lemma_count_df[lemma_count_df.majority_tag == 'NOUN']))
print('Verb lemmas:', len(lemma_count_df[lemma_count_df.majority_tag == 'VERB']))


# In[ ]:


lemma_count_df[['nv_cosine_similarity', 'n_variation', 'v_variation']] =   lemma_count_df.apply(lambda row: embedder.get_contextual_nv_similarity(row.lemma, method="bert"),
                       axis=1, result_type="expand")


# In[ ]:


lemma_count_df[['lemma', 'noun_count', 'verb_count', 'majority_tag', 'nv_cosine_similarity', 'n_variation', 'v_variation']]   .sort_values('nv_cosine_similarity').head(8)


# In[ ]:


lemma_count_df[['lemma', 'noun_count', 'verb_count', 'majority_tag', 'nv_cosine_similarity', 'n_variation', 'v_variation']]   .sort_values('nv_cosine_similarity', ascending=False).head(8)


# ## Difference in similarity when base is noun vs verb

# In[ ]:


plot = sns.distplot(lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].nv_cosine_similarity, label='Base=N')
plot = sns.distplot(lemma_count_df[lemma_count_df.majority_tag == 'VERB'].nv_cosine_similarity, label='Base=V')
plt.legend()
plot.set(title="Average Cosine Similarity between Noun/Verb Usage",
         xlabel="Cosine Similarity", ylabel="Count")
plt.show()


# In[ ]:


print('Mean cosine similarity when Base=N:', np.mean(lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].nv_cosine_similarity))
print('Mean cosine similarity when Base=V:', np.mean(lemma_count_df[lemma_count_df.majority_tag == 'VERB'].nv_cosine_similarity))


# In[ ]:


# T-test of difference in mean
scipy.stats.ttest_ind(lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].nv_cosine_similarity,
                      lemma_count_df[lemma_count_df.majority_tag == 'VERB'].nv_cosine_similarity)


# ## Difference in variation between noun and verb

# In[ ]:


print('Mean noun variation:', np.mean(lemma_count_df.n_variation))
print('Mean verb variation:', np.mean(lemma_count_df.v_variation))


# In[ ]:


# Paired t-test for difference
scipy.stats.ttest_rel(lemma_count_df.n_variation, lemma_count_df.v_variation)


# ## Difference in variation between majority and minority class

# In[ ]:


majority_variation = np.where(lemma_count_df.majority_tag == 'NOUN', lemma_count_df.n_variation, lemma_count_df.v_variation)
minority_variation = np.where(lemma_count_df.majority_tag == 'NOUN', lemma_count_df.v_variation, lemma_count_df.n_variation)
plot = sns.distplot(majority_variation, label='Majority')
plot = sns.distplot(minority_variation, label='Minority')
plt.legend()
plot.set(title="Semantic variation within majority and minority POS class",
         xlabel="Standard deviation", ylabel="Density")
plt.show()


# In[ ]:


print('Mean majority variation:', np.mean(majority_variation))
print('Mean minority variation:', np.mean(minority_variation))


# In[ ]:


# Paired t-test for difference
scipy.stats.ttest_rel(majority_variation, minority_variation)


# ## MTurk correlation

# In[ ]:


annotation_df = pd.read_csv('../data/annotations/myself_plus_mturk.csv')


# In[ ]:


annotation_df = pd.merge(annotation_df, lemma_count_df, on='lemma')


# In[ ]:


scipy.stats.spearmanr(annotation_df.mean_score, annotation_df.nv_cosine_similarity)

