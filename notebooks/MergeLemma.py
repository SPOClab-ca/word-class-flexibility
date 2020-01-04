#!/usr/bin/env python
# coding: utf-8

# # Union-find Lemma Merging Experiments
# 
# Play with the following idea: take all the set of all words that have lemma A, and the set of all words that have lemma B, and if there is any overlap between the two sets, then we merge A and B into the same lemma.

# In[1]:


import sys
sys.path.append('../')

import glob
from disjoint_set import DisjointSet

import src.corpus

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


FRENCH_UD_FILES = [f for f in glob.glob('../data/ud_all/ud-treebanks-v2.5/**/*.conllu') if 'French' in f]


# In[3]:


FRENCH_UD_FILES[:3]


# ## Construct sets that share lemmas

# In[ ]:


corpus = src.corpus.POSCorpus.create_from_ud(data_file_list=FRENCH_UD_FILES)


# In[ ]:


ds = DisjointSet()
for sentence in corpus.sentences:
  for token in sentence:
    ds.union(token['word'], token['lemma'])


# In[ ]:


print(ds.find('voyage'))
print(ds.find('voyages'))
print(ds.find('voyager'))
print(ds.find('voyagent'))


# In[ ]:


print(ds.find('chant'))
print(ds.find('chants'))
print(ds.find('chanter'))
print(ds.find('chante'))
print(ds.find('chantant'))

