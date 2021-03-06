#!/usr/bin/env python
# coding: utf-8

# # Compare Embeddings
# 
# Notebook to compare different embedding methods against MTurk labels to see what corresponds most with human judgements of semantic similarity

# In[1]:


import sys
sys.path.append('../')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats

import src.corpus
import src.semantic_embedding


# ## Parse the corpus

# In[2]:


BNC_FILE = "../data/bnc/bnc.pkl"
corpus = src.corpus.POSCorpus.create_from_bnc_pickled(data_file_path=BNC_FILE)


# In[3]:


annotation_df = pd.read_csv('../data/annotations/myself_plus_mturk.csv')
relevant_lemmas = annotation_df.lemma.tolist()
annotation_df.head()


# ## Filter sentences containing lemmas we care about

# In[4]:


sentences_with_relevant_lemmas = []
for sentence in corpus.sentences:
  for tok in sentence:
    if tok['lemma'] in relevant_lemmas:
      sentences_with_relevant_lemmas.append(sentence)
      break
sentences_with_relevant_lemmas = sentences_with_relevant_lemmas[:50000]


# In[ ]:





# ## Embedder method: ELMo
embedder = src.semantic_embedding.SemanticEmbedding(sentences_with_relevant_lemmas)
embedder.init_elmo(layer=0)
annotation_df['nv_cosine_similarity'] = \
  annotation_df.apply(lambda row: embedder.get_elmo_nv_similarity(row.lemma), axis=1)
# ## Embedder method: BERT

# In[5]:


layer = 12
embedder = src.semantic_embedding.SemanticEmbedding(sentences_with_relevant_lemmas)
embedder.init_bert(model_name='bert-base-multilingual-cased', layer=layer)
annotation_df[['nv_cosine_similarity', 'n_variation', 'v_variation']] =   annotation_df.apply(lambda row: embedder.get_contextual_nv_similarity(row.lemma, method="bert"),
                       axis=1, result_type="expand")


# ## Embedder method: GloVe
embedder = src.semantic_embedding.SemanticEmbedding(sentences_with_relevant_lemmas)
embedder.init_glove()
annotation_df['nv_cosine_similarity'] = annotation_df.apply(
  lambda row: embedder.get_glove_nv_similarity(row.lemma, context=0, include_self=True),
  axis=1
)
# ## Run NV similarity

# In[10]:


corr = scipy.stats.spearmanr(annotation_df.mean_score, annotation_df.nv_cosine_similarity)[0]


# In[11]:


plot = sns.boxplot(annotation_df.mean_score, annotation_df.nv_cosine_similarity)
plot.set_title('BERT layer %d, corr = %0.9f' % (layer, corr))
#plot.get_figure().savefig('figs/bert_%d.png' % layer)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




