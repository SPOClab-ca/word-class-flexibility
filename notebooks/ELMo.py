#!/usr/bin/env python
# coding: utf-8

# # ELMo Contextual Embeddings
# 
# In this notebook, we use contextual embeddings from ELMo to study semantic change of conversion.

# In[1]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import allennlp.commands.elmo
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.decomposition
import sklearn.metrics
import tqdm
import random

import src.corpus

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Parse the corpus

# In[2]:


#UD_FILE = "../data/en_ewt-ud-train.conllu"
#corpus = src.corpus.POSCorpus.create_from_ud(data_file_path=UD_FILE)

BNC_FILE = "../data/bnc/bnc.pkl"
corpus = src.corpus.POSCorpus.create_from_bnc_pickled(data_file_path=BNC_FILE)


# In[3]:


# Define the two POS (NOUN, VERB, ADJ) to compare for the rest of the analysis
POS1 = 'NOUN'
POS2 = 'VERB'


# ## Run ELMo on random part of the corpus

# In[4]:


# Take only 1M words out of 4M to make it run faster
SAMPLE_PROPORTION = 0.25
random.seed(12345)
random_indices = random.sample(range(len(corpus.sentences)), int(SAMPLE_PROPORTION * len(corpus.sentences)))

sampled_sentences = []
for ix in random_indices:
  sampled_sentences.append(corpus.sentences[ix])


# In[5]:


elmo = allennlp.commands.elmo.ElmoEmbedder(cuda_device=0)
data_as_tokens = [[t['word'] for t in sentence] for sentence in sampled_sentences]

BATCH_SIZE = 64
elmo_embeddings = []
for ix in tqdm.tqdm(range(0, len(data_as_tokens), BATCH_SIZE)):
  batch = data_as_tokens[ix : ix+BATCH_SIZE]
  batch_embeddings = elmo.embed_batch(batch)
  # Only take embeddings from last ELMo layer
  batch_embeddings = [x[-1] for x in batch_embeddings]
  elmo_embeddings.extend(batch_embeddings)


# ## ELMo embeddings of instances of a fixed lemma

# In[6]:


def get_elmo_embeddings_for_lemma(lemma):
  pos1_embeddings = []
  pos2_embeddings = []

  for sentence_ix in range(len(sampled_sentences)):
    token_list = sampled_sentences[sentence_ix]
    embeddings = elmo_embeddings[sentence_ix]
    for i in range(len(token_list)):
      if token_list[i]['lemma'] == lemma:
        if token_list[i]['pos'] == POS1:
          pos1_embeddings.append(embeddings[i])
        elif token_list[i]['pos'] == POS2:
          pos2_embeddings.append(embeddings[i])

  pos1_embeddings = np.vstack(pos1_embeddings)
  pos2_embeddings = np.vstack(pos2_embeddings)
  return pos1_embeddings, pos2_embeddings


# In[7]:


FIXED_LEMMA = "use"
pos1_embeddings, pos2_embeddings = get_elmo_embeddings_for_lemma(FIXED_LEMMA)
print("%s instances: %d" % (POS1, pos1_embeddings.shape[0]))
print("%s instances: %d" % (POS2, pos2_embeddings.shape[0]))


# ## Apply PCA and plot

# In[8]:


pca = sklearn.decomposition.PCA(n_components=2)
all_embeddings = pca.fit_transform(np.vstack([pos1_embeddings, pos2_embeddings]))
all_embeddings_df = pd.DataFrame({'x0': all_embeddings[:,0], 'x1': all_embeddings[:,1]})
all_embeddings_df['pos'] = [POS1] * len(pos1_embeddings) + [POS2] * len(pos2_embeddings)


# In[9]:


plot = sns.scatterplot(data=all_embeddings_df, x='x0', y='x1', hue='pos')
plot.set(title="ELMo embeddings for lemma: '%s'" % FIXED_LEMMA)
plt.show()


# ## Utility to inspect what ELMo is capturing

# In[10]:


num_printed = 0
for sentence_ix in range(len(sampled_sentences)):
  token_list = sampled_sentences[sentence_ix]
  embeddings = elmo_embeddings[sentence_ix]
  for i in range(len(token_list)):
    if token_list[i]['lemma'] == FIXED_LEMMA:
      v = pca.transform(embeddings[i][np.newaxis, :])[0]
      if v[1] < -10: # <- Put whatever condition here
        print(v)
        print(' '.join([t['word'] for t in token_list]))
        print()
        num_printed += 1
  if num_printed > 10:
    break


# ## Cosine similarity between POS1 and POS2 usages

# In[11]:


lemma_count_df = corpus.get_per_lemma_stats()

# Filter: must have at least [x] POS1 and [x] POS2 usages
lemma_count_df = lemma_count_df[(lemma_count_df['pos1_count'] >= 30) & (lemma_count_df['pos2_count'] >= 30)]
lemma_count_df = lemma_count_df.sort_values('total_count', ascending=False)
print('Remaining lemmas:', len(lemma_count_df))


# In[12]:


def get_pos_cosine_similarity(row):
  pos1_embeddings, pos2_embeddings = get_elmo_embeddings_for_lemma(row.lemma)
  
  avg_pos1_embedding = np.mean(pos1_embeddings, axis=0)
  avg_pos2_embedding = np.mean(pos2_embeddings, axis=0)

  return float(sklearn.metrics.pairwise.cosine_similarity(avg_pos1_embedding[np.newaxis,:], avg_pos2_embedding[np.newaxis,:]))

lemma_count_df['pos_cosine_similarity'] = lemma_count_df.apply(get_pos_cosine_similarity, axis=1)


# In[13]:


lemma_count_df[['lemma', 'pos1_count', 'pos2_count', 'majority_tag', 'pos_cosine_similarity']]   .sort_values('pos_cosine_similarity').head(8)


# In[14]:


lemma_count_df[['lemma', 'pos1_count', 'pos2_count', 'majority_tag', 'pos_cosine_similarity']]   .sort_values('pos_cosine_similarity', ascending=False).head(8)


# In[15]:


plot = sns.distplot(lemma_count_df[lemma_count_df.majority_tag == POS1].pos_cosine_similarity, label='Base=%s' % POS1)
plot = sns.distplot(lemma_count_df[lemma_count_df.majority_tag == POS2].pos_cosine_similarity, label='Base=%s' % POS2)
plt.legend()
plot.set(title="Average Cosine Similarity between %s/%s Usage" % (POS1, POS2),
         xlabel="Cosine Similarity", ylabel="Count")
plt.show()


# In[16]:


# T-test of difference in mean
import scipy.stats
scipy.stats.ttest_ind(lemma_count_df[lemma_count_df.majority_tag == POS1].pos_cosine_similarity,
                      lemma_count_df[lemma_count_df.majority_tag == POS2].pos_cosine_similarity)

