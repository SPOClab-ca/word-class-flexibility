#!/usr/bin/env python
# coding: utf-8

# # Basic UD Part-of-speech Analysis
# 
# Fairly basic analysis involving POS tags on some Universal Dependencies corpora

# In[1]:


from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# In[5]:


plt.figure(figsize=(12, 6))
plt.bar(pos_counts.keys(), pos_counts.values())


# ## Gather usages of each lemma

# In[6]:


# {lemma -> (POS, word, sentence)}
lemma_forms = defaultdict(list)

for token_list in data:
  sentence = ' '.join([t['form'] for t in token_list])
  for token in token_list:
    pos_tag = token['upostag']
    lemma = token['lemma']
    word = token['form']
    lemma_forms[lemma].append((pos_tag, word, sentence))


# ## Noun/Verb statistics for each lemma

# In[7]:


lemma_count_df = []
for lemma, lemma_occurrences in lemma_forms.items():
  noun_count = len([word for (pos, word, _) in lemma_occurrences if pos == 'NOUN'])
  verb_count = len([word for (pos, word, _) in lemma_occurrences if pos == 'VERB'])
  lemma_count_df.append({'lemma': lemma, 'noun_count': noun_count, 'verb_count': verb_count})
lemma_count_df = pd.DataFrame(lemma_count_df)


# In[8]:


# Filter and compute minority count and ratio
lemma_count_df = lemma_count_df[lemma_count_df['noun_count'] + lemma_count_df['verb_count'] > 0]
lemma_count_df['majority_tag'] = np.where(lemma_count_df['noun_count'] >= lemma_count_df['verb_count'], 'NOUN', 'VERB')
lemma_count_df['total_count'] = lemma_count_df[['noun_count', 'verb_count']].sum(axis=1)
lemma_count_df['minority_count'] = lemma_count_df[['noun_count', 'verb_count']].min(axis=1)
lemma_count_df['minority_ratio'] = lemma_count_df['minority_count'] / lemma_count_df['total_count']
lemma_count_df['is_flexible'] = lemma_count_df['minority_ratio'] > 0.05


# In[9]:


lemma_count_df.sort_values('total_count', ascending=False).head(20)


# ## Distribution of lemmas

# In[10]:


plt.figure(figsize=(15, 5))
lemma_count_df['total_count'].hist(bins=range(0, 60))


# ## Syntax flexibility metrics

# In[11]:


# Only consider lemmas with at least 5 usages
lemma_count_df = lemma_count_df[lemma_count_df['total_count'] >= 5].sort_values('total_count', ascending=False)
noun_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'NOUN'])
verb_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'VERB'])
noun_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])]) / noun_lemmas
verb_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])]) / verb_lemmas


# In[12]:


noun_flexibility


# In[13]:


verb_flexibility


# In[14]:


# Top flexible nouns
lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])].head(10)


# In[15]:


# Examples of inflexible nouns
lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (~lemma_count_df['is_flexible'])].head(10)


# In[16]:


# Examples of flexible verbs
lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])].head(10)


# In[17]:


# Examples of inflexible verbs
lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (~lemma_count_df['is_flexible'])].head(10)

