#!/usr/bin/env python
# coding: utf-8

# # Compare Embeddings
# 
# Notebook to compare different embedding methods against MTurk labels to see what corresponds most with human judgements of semantic similarity

# In[1]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats

import src.corpus
import src.semantic_embedding

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Parse the corpus

# In[2]:


BNC_FILE = "../data/bnc/bnc.pkl"
corpus = src.corpus.POSCorpus.create_from_bnc_pickled(data_file_path=BNC_FILE)


# In[3]:


annotation_df = pd.read_csv('../data/annotations/myself_plus_mturk.csv')
relevant_lemmas = annotation_df.lemma.tolist()
annotation_df.head()


# ## Run ELMo, but not the whole thing

# In[10]:


sentences_with_relevant_lemmas = []
for sentence in corpus.sentences:
  for tok in sentence:
    if tok['lemma'] in relevant_lemmas:
      sentences_with_relevant_lemmas.append(sentence)
      break


# In[11]:


embedder = src.semantic_embedding.SemanticEmbedding(sentences_with_relevant_lemmas)


# ## Run ELMo NV similarity

# In[12]:


annotation_df['nv_cosine_similarity'] =   annotation_df.apply(lambda row: embedder.get_nv_cosine_similarity(row.lemma), axis=1)


# In[13]:


sns.boxplot(annotation_df.mean_score, annotation_df.nv_cosine_similarity)


# In[14]:


scipy.stats.pearsonr(annotation_df.mean_score, annotation_df.nv_cosine_similarity)

