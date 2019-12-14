#!/usr/bin/env python
# coding: utf-8

# # Multilingual UD
# 
# Compute noun/verb frequency-based statistics for all languages in UD

# In[1]:


import glob
import os
from collections import defaultdict


# In[2]:


# Group all treebanks by language
UD_PATH = '../data/ud_all/ud-treebanks-v2.5/'
ud_files = defaultdict(list)

for ud_corpus_name in os.listdir(UD_PATH):
  language_name = ud_corpus_name[3:].split('-')[0].replace('_', ' ')
  for conllu_file in glob.glob(UD_PATH + ud_corpus_name + '/*.conllu'):
    #conllu_file_name = os.path.basename(conllu_file)
    #language_code = conllu_file_name.split('_')[0]
    ud_files[language_name].append(conllu_file)


# In[3]:


ud_files['French']

