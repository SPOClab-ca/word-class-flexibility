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
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

import src.corpus

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


FRENCH_UD_FILES = [f for f in glob.glob('../data/ud_all/ud-treebanks-v2.5/**/*.conllu') if 'French' in f]


# In[3]:


FRENCH_UD_FILES[:3]


# ## Construct sets that share lemmas

# In[4]:


corpus = src.corpus.POSCorpus.create_from_ud(data_file_list=FRENCH_UD_FILES)


# In[5]:


# Helper iterate to return N/V words and lemmas in corpus, lowercased
def iterate_words(corpus):
  for sentence in corpus.sentences:
    for token in sentence:
      if token['pos'] in ['NOUN', 'VERB']:
        yield token['word'].lower(), token['lemma'].lower()


# In[6]:


ds = DisjointSet()
for word, lemma in iterate_words(corpus):
  ds.union(word, lemma)


# In[7]:


print(ds.find('voyage'))
print(ds.find('voyages'))
print(ds.find('voyager'))
print(ds.find('voyagent'))


# In[8]:


print(ds.find('chant'))
print(ds.find('chants'))
print(ds.find('chanter'))
print(ds.find('chante'))
print(ds.find('chantant'))


# ## Group words that share the same lemma

# In[9]:


lemma_counter = Counter()
for _, lemma in iterate_words(corpus):
  lemma_counter[lemma] += 1


# In[10]:


lemma_groups = defaultdict(set)
for word, lemma in iterate_words(corpus):
  lemma_groups[ds.find(word)].add(word)


# In[33]:


lemma_groups[ds.find('mourir')]


# In[12]:


# Name of the group is the most frequent lemma in the group
def get_name_for_group(word):
  maxn, maxw = 0, None
  for w in lemma_groups[ds.find(word)]:
    if lemma_counter[w] > maxn:
      maxn = lemma_counter[w]
      maxw = w
  return maxw


# In[13]:


print(get_name_for_group('parle'))
print(get_name_for_group('font'))


# ## NV flexibility stats
# 
# Modified from `corpus.py`

# In[14]:


flexibility_threshold = 0.05
lemma_forms = defaultdict(list)
for sentence in corpus.sentences:
  for token in sentence:
    lemma = token['lemma'].lower()
    word = token['word'].lower()
    pos = token['pos']
    lemma_forms[ds.find(lemma)].append((pos, word))

lemma_count_df = []
for lemma, lemma_occurrences in lemma_forms.items():
  noun_count = len([word for (pos, word) in lemma_occurrences if pos == 'NOUN'])
  verb_count = len([word for (pos, word) in lemma_occurrences if pos == 'VERB'])
  lemma_count_df.append({'lemma': get_name_for_group(lemma), 'noun_count': noun_count, 'verb_count': verb_count})
lemma_count_df = pd.DataFrame(lemma_count_df)

lemma_count_df = lemma_count_df[lemma_count_df['noun_count'] + lemma_count_df['verb_count'] > 0]
lemma_count_df['majority_tag'] = np.where(lemma_count_df['noun_count'] >= lemma_count_df['verb_count'], 'NOUN', 'VERB')
lemma_count_df['total_count'] = lemma_count_df[['noun_count', 'verb_count']].sum(axis=1)
lemma_count_df['minority_count'] = lemma_count_df[['noun_count', 'verb_count']].min(axis=1)
lemma_count_df['minority_ratio'] = lemma_count_df['minority_count'] / lemma_count_df['total_count']
lemma_count_df['is_flexible'] = lemma_count_df['minority_ratio'] > flexibility_threshold
lemma_count_df = lemma_count_df[lemma_count_df['total_count'] >= 10].sort_values('total_count', ascending=False)


# In[15]:


lemma_count_df.head(20)


# ## Syntax flexibility metrics

# In[16]:


noun_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'NOUN'])
verb_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'VERB'])
noun_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])]) / noun_lemmas
verb_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])]) / verb_lemmas


# In[17]:


print('Noun Flexibility = P(flexible | noun):', noun_flexibility)


# In[18]:


print('Verb Flexibility = P(flexible | verb):', verb_flexibility)


# In[19]:


# Compute ratio of flexible words that are nouns, to compare with Balteiro (2007)
num_flexible = len(lemma_count_df[lemma_count_df['is_flexible']])
num_flexible_nouns = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & lemma_count_df['is_flexible']])
print("Flexibility Asymmetry = P(noun | flexible):", num_flexible_nouns / num_flexible)


# ## Show Examples

# In[20]:


# Top flexible nouns
lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])].head(10)


# In[21]:


# Examples of inflexible nouns
lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (~lemma_count_df['is_flexible'])].head(10)


# In[22]:


# Examples of flexible verbs
lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])].head(10)


# In[23]:


# Examples of inflexible verbs
lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (~lemma_count_df['is_flexible'])].head(10)

