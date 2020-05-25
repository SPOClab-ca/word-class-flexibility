#!/usr/bin/env python
# coding: utf-8

# # Factors of Flexibility
# 
# Investigate what factors influence whether a word is flexible or not.

# In[1]:


import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import src.corpus
import src.partial

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Parse the corpus

# In[2]:


data_file = "../data/wiki/processed/en.pkl"

corpus = src.corpus.POSCorpus.create_from_pickle(data_file_path=data_file)


# In[3]:


lemma_count_df = corpus.get_per_lemma_stats()
lemma_count_df = lemma_count_df[lemma_count_df.total_count >= 100]
lemma_count_df.sort_values('total_count', ascending=False).head(20)


# ## LR to predict flexibility

# In[4]:


lemma_count_df['log_freq'] = np.log(lemma_count_df.total_count)
lemma_count_df['length'] = lemma_count_df.lemma.apply(lambda x: len(x))


# In[5]:


import statsmodels.discrete.discrete_model
model = statsmodels.discrete.discrete_model.Logit(
  lemma_count_df.is_flexible,
  pd.get_dummies(lemma_count_df[['majority_tag', 'log_freq', 'length']], drop_first=True)
)
lr = model.fit()


# In[6]:


lr.summary()


# In[7]:


lr.params.tolist()


# ## Partial correlation

# In[8]:


partials = src.partial.calculate_partial_correlation(pd.get_dummies(lemma_count_df[['majority_tag', 'log_freq', 'length', 'is_flexible']], drop_first=True))
print(partials['is_flexible']['log_freq'])
print(partials['is_flexible']['length'])
print(partials['is_flexible']['majority_tag_VERB'])
partials

