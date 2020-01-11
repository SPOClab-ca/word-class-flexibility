#!/usr/bin/env python
# coding: utf-8

# # Multilingual UD
# 
# Compute noun/verb frequency-based statistics for all languages in UD

# In[1]:


import sys
sys.path.append('../')

import glob
import os
from collections import defaultdict
import pandas as pd
import multiprocessing as mp

import src.corpus

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


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


ud_files['French'][:5]


# ## All UD files in one language

# In[4]:


corpus = src.corpus.POSCorpus.create_from_ud(data_file_list=ud_files['French'])


# In[5]:


lemma_count_df = corpus.get_lemma_stats_merge_method()
lemma_count_df.sort_values('total_count', ascending=False).head(10)


# In[6]:


total_tokens = sum([len(sentence) for sentence in corpus.sentences])
print('Total tokens:', total_tokens)


# In[7]:


# Only consider lemmas with at least 10 usages
lemma_count_df = lemma_count_df[lemma_count_df['total_count'] >= 10].sort_values('total_count', ascending=False)
noun_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'NOUN'])
verb_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'VERB'])
noun_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])]) / noun_lemmas
verb_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])]) / verb_lemmas


# In[8]:


print('Noun lemmas with >= 10 usages:', noun_lemmas)
print('Verb lemmas with >= 10 usages:', verb_lemmas)


# In[9]:


print('Noun Flexibility = P(flexible | noun):', noun_flexibility)
print('Verb Flexibility = P(flexible | verb):', verb_flexibility)


# ## Loop over all languages

# In[ ]:


def process_ud_language(args):
  language_name, language_ud_list = args
  print('Processing:', language_name)
  
  corpus = src.corpus.POSCorpus.create_from_ud(data_file_list=ud_files[language_name])
  if len(corpus.sentences) == 0: return None
  total_tokens = sum([len(sentence) for sentence in corpus.sentences])
  lemma_count_df = corpus.get_lemma_stats_merge_method()
  lemma_count_df = lemma_count_df[lemma_count_df['total_count'] >= 10].sort_values('total_count', ascending=False)
  noun_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'NOUN'])
  verb_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'VERB'])
  if noun_lemmas == 0 or verb_lemmas == 0: return None
  noun_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])]) / noun_lemmas
  verb_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])]) / verb_lemmas
  
  return pd.Series({
    'language': language_name,
    'tokens': total_tokens,
    'noun_lemmas': noun_lemmas,
    'verb_lemmas': verb_lemmas,
    'noun_flexibility': noun_flexibility,
    'verb_flexibility': verb_flexibility,
  })

pool = mp.Pool()
results = pool.map(process_ud_language, ud_files.items())
results = [r for r in results if r is not None]
all_language_stats = pd.DataFrame(results)


# In[ ]:


all_language_stats = all_language_stats.sort_values('tokens', ascending=False)
all_language_stats


# In[ ]:


all_language_stats.to_csv('multi-language-ud.csv', index=False)


# In[ ]:


all_language_stats[(all_language_stats.noun_flexibility > 0.05) & (all_language_stats.verb_flexibility > 0.05)]

