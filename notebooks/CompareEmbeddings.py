#!/usr/bin/env python
# coding: utf-8

# # Compare Embeddings
# 
# Notebook to compare different embedding methods against MTurk labels to see what corresponds most with human judgements of semantic similarity

# In[38]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import allennlp.commands.elmo
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import random

import sklearn.metrics
import scipy.stats

import src.corpus

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Parse the corpus

# In[23]:


BNC_FILE = "../data/bnc/bnc.pkl"
corpus = src.corpus.POSCorpus.create_from_bnc_pickled(data_file_path=BNC_FILE)


# In[24]:


annotation_df = pd.read_csv('../data/annotations/myself_plus_mturk.csv')
relevant_lemmas = annotation_df.lemma.tolist()
annotation_df.head()


# ## Run ELMo, but not the whole thing

# In[26]:


sentences_with_relevant_lemmas = []
for sentence in corpus.sentences:
  for tok in sentence:
    if tok['lemma'] in relevant_lemmas:
      sentences_with_relevant_lemmas.append(sentence)
      break


# In[27]:


elmo = allennlp.commands.elmo.ElmoEmbedder(cuda_device=0)
data_as_tokens = [[t['word'] for t in sentence] for sentence in sentences_with_relevant_lemmas]

BATCH_SIZE = 64
elmo_embeddings = []
for ix in tqdm.tqdm(range(0, len(data_as_tokens), BATCH_SIZE)):
  batch = data_as_tokens[ix : ix+BATCH_SIZE]
  batch_embeddings = elmo.embed_batch(batch)
  # Only take embeddings from last ELMo layer
  batch_embeddings = [x[-1] for x in batch_embeddings]
  elmo_embeddings.extend(batch_embeddings)


# ## Copy over from ELMo.ipynb

# In[31]:


def get_elmo_embeddings_for_lemma(lemma):
  noun_embeddings = []
  verb_embeddings = []

  for sentence_ix in range(len(sentences_with_relevant_lemmas)):
    token_list = sentences_with_relevant_lemmas[sentence_ix]
    embeddings = elmo_embeddings[sentence_ix]
    for i in range(len(token_list)):
      if token_list[i]['lemma'] == lemma:
        if token_list[i]['pos'] == 'NOUN':
          noun_embeddings.append(embeddings[i])
        elif token_list[i]['pos'] == 'VERB':
          verb_embeddings.append(embeddings[i])

  noun_embeddings = np.vstack(noun_embeddings)
  verb_embeddings = np.vstack(verb_embeddings)
  return noun_embeddings, verb_embeddings

def get_nv_cosine_similarity(row):
  noun_embeddings, verb_embeddings = get_elmo_embeddings_for_lemma(row.lemma)
  
  avg_noun_embedding = np.mean(noun_embeddings, axis=0)
  avg_verb_embedding = np.mean(verb_embeddings, axis=0)

  return float(sklearn.metrics.pairwise.cosine_similarity(avg_noun_embedding[np.newaxis,:], avg_verb_embedding[np.newaxis,:]))

annotation_df['nv_cosine_similarity'] = annotation_df.apply(get_nv_cosine_similarity, axis=1)


# In[36]:


sns.boxplot(annotation_df.mean_score, annotation_df.nv_cosine_similarity)


# In[49]:


scipy.stats.pearsonr(annotation_df.mean_score, annotation_df.nv_cosine_similarity)

