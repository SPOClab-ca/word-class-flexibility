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

import src.corpus
import src.semantic_embedding

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Parse the corpus

# In[2]:


#UD_FILE = "../data/en_ewt-ud-train.conllu"
#corpus = src.corpus.POSCorpus.create_from_ud(data_file_path=UD_FILE)

BNC_FILE = "../data/bnc/bnc.pkl"
corpus = src.corpus.POSCorpus.create_from_bnc_pickled(data_file_path=BNC_FILE)


# ## Compute embeddings on random part of the corpus

# In[ ]:


# Take only 1M words out of 4M to make it run faster
SAMPLE_PROPORTION = 1.0
random.seed(12345)
random_indices = random.sample(range(len(corpus.sentences)), int(SAMPLE_PROPORTION * len(corpus.sentences)))

sampled_sentences = []
for ix in random_indices:
  sampled_sentences.append(corpus.sentences[ix])
  
embedder = src.semantic_embedding.SemanticEmbedding(sampled_sentences)
embedder.init_bert(layer=12)


# ## Compute embeddings of instances of a fixed lemma

# In[39]:


FIXED_LEMMA = "store"
noun_embeddings, verb_embeddings = embedder.get_bert_embeddings_for_lemma(FIXED_LEMMA)
print("Noun instances:", noun_embeddings.shape[0])
print("Verb instances:", verb_embeddings.shape[0])


# ## Apply PCA and plot

# In[40]:


pca = sklearn.decomposition.PCA(n_components=2)
all_embeddings = pca.fit_transform(np.vstack([noun_embeddings, verb_embeddings]))
all_embeddings_df = pd.DataFrame({'x0': all_embeddings[:,0], 'x1': all_embeddings[:,1]})
all_embeddings_df['pos'] = ['noun'] * len(noun_embeddings) + ['verb'] * len(verb_embeddings)


# In[41]:


plot = sns.scatterplot(data=all_embeddings_df, x='x0', y='x1', hue='pos')
plot.set(title="Contextual embeddings for lemma: '%s'" % FIXED_LEMMA)
plt.show()


# ## Utility to inspect what it's capturing
num_printed = 0
for sentence_ix in range(len(sampled_sentences)):
  token_list = sampled_sentences[sentence_ix]
  embeddings = embedder.bert_embeddings[sentence_ix]
  for i in range(len(token_list)):
    if token_list[i]['lemma'] == FIXED_LEMMA:
      v = pca.transform(embeddings[i][np.newaxis, :])[0]
      if 1 < v[0] < 2: # <- Put whatever condition here
        print(v)
        print(' '.join([t['word'] for t in token_list]))
        print()
        num_printed += 1
  if num_printed > 10:
    break
# ## Cosine similarity between noun and verb usages

# In[27]:


lemma_count_df = corpus.get_per_lemma_stats()

# Filter: must have at least [x] noun and [x] verb usages
lemma_count_df = lemma_count_df[(lemma_count_df['noun_count'] >= 80) & (lemma_count_df['verb_count'] >= 80)]
lemma_count_df = lemma_count_df.sort_values('total_count', ascending=False)
lemma_count_df = lemma_count_df[~lemma_count_df.lemma.isin(['go', 'will', 'may'])]
print('Remaining lemmas:', len(lemma_count_df))


# In[51]:


lemma_count_df[['nv_cosine_similarity', 'n_variation', 'v_variation']] =   lemma_count_df.apply(lambda row: embedder.get_contextual_nv_similarity(row.lemma, method="bert"),
                       axis=1, result_type="expand")


# In[54]:


lemma_count_df[['lemma', 'noun_count', 'verb_count', 'majority_tag', 'nv_cosine_similarity']]   .sort_values('nv_cosine_similarity').head(8)


# In[55]:


lemma_count_df[['lemma', 'noun_count', 'verb_count', 'majority_tag', 'nv_cosine_similarity']]   .sort_values('nv_cosine_similarity', ascending=False).head(8)


# In[56]:


plot = sns.distplot(lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].nv_cosine_similarity, label='Base=N')
plot = sns.distplot(lemma_count_df[lemma_count_df.majority_tag == 'VERB'].nv_cosine_similarity, label='Base=V')
plt.legend()
plot.set(title="Average Cosine Similarity between Noun/Verb Usage",
         xlabel="Cosine Similarity", ylabel="Count")
plt.show()


# In[57]:


# T-test of difference in mean
import scipy.stats
scipy.stats.ttest_ind(lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].nv_cosine_similarity,
                      lemma_count_df[lemma_count_df.majority_tag == 'VERB'].nv_cosine_similarity)
