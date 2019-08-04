#!/usr/bin/env python
# coding: utf-8

# # Basic UD Part-of-speech Analysis
# 
# Fairly basic analysis involving POS tags on some Universal Dependencies corpora

# In[1]:


import matplotlib.pyplot as plt
from collections import defaultdict
import conllu

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Read the CoNLL-U file

# In[2]:


UD_FILE = "../data/en_partut-ud-train.conllu"

with open(UD_FILE, "r", encoding="utf-8") as data_file:
  data = data_file.read()
  data = conllu.parse(data)


# In[3]:


data[:3]


# ## POS counts

# In[4]:


pos_counts = defaultdict(int)

for token_list in data:
  for token in token_list:
    pos_tag = token['upostag']
    pos_counts[pos_tag] += 1

pos_counts


# In[5]:


plt.figure(figsize=(12, 6))
plt.bar(pos_counts.keys(), pos_counts.values())

