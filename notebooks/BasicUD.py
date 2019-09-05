#!/usr/bin/env python
# coding: utf-8

# # Basic UD Part-of-speech Analysis
# 
# Fairly basic analysis involving POS tags on some Universal Dependencies corpora

# In[1]:


import sys
sys.path.append('../')

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import src.corpus

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Parse the corpus

# In[2]:


#UD_FILE = "../data/zh_gsd-ud-train.conllu"
#UD_FILE = "../data/en_ewt-ud-train.conllu"
#UD_FILE = "../data/ja_gsd-ud-train.conllu"

BNC_FILE = "../data/bnc/bnc.pkl"

corpus = src.corpus.POSCorpus.create_from_bnc_pickled(data_file_path=BNC_FILE)


# In[3]:


# Define the two POS (NOUN, VERB, ADJ) to compare for the rest of the analysis
POS1 = 'NOUN'
POS2 = 'VERB'


# ## POS counts

# In[4]:


pos_counts = defaultdict(int)

for sentence in corpus.sentences:
  for token in sentence:
    pos_tag = token['pos']
    if pos_tag:
      pos_counts[pos_tag] += 1


# In[5]:


plt.figure(figsize=(12, 6))
plt.bar(pos_counts.keys(), pos_counts.values())


# In[6]:


lemma_count_df = corpus.get_per_lemma_stats(POS1=POS1, POS2=POS2)
lemma_count_df.sort_values('total_count', ascending=False).head(20)


# ## Distribution of lemmas

# In[7]:


plt.figure(figsize=(15, 5))
lemma_count_df['total_count'].hist(bins=range(0, 60))


# ## Syntax flexibility metrics

# In[8]:


# Only consider lemmas with at least 5 usages
lemma_count_df = lemma_count_df[lemma_count_df['total_count'] >= 5].sort_values('total_count', ascending=False)
pos1_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == POS1])
pos2_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == POS2])
pos1_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == POS1) & (lemma_count_df['is_flexible'])]) / pos1_lemmas
pos2_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == POS2) & (lemma_count_df['is_flexible'])]) / pos2_lemmas


# In[9]:


print('%s Flexibility = P(flexible | %s): %0.9f' % (POS1, POS1, pos1_flexibility))


# In[10]:


print('%s Flexibility = P(flexible | %s): %0.9f' % (POS2, POS2, pos2_flexibility))


# In[11]:


# Compute ratio of flexible words that are POS1, to compare with Balteiro (2007)
num_flexible = len(lemma_count_df[lemma_count_df['is_flexible']])
num_flexible_pos1 = len(lemma_count_df[(lemma_count_df['majority_tag'] == POS1) & lemma_count_df['is_flexible']])
print("Flexibility Asymmetry = P(%s | flexible): %0.9f" % (POS1, num_flexible_pos1 / num_flexible))


# In[12]:


flexible_df = lemma_count_df[lemma_count_df.is_flexible]
dplot = sns.distplot(flexible_df.pos1_count / flexible_df.total_count, bins=20)
dplot.set(xlabel='POS1 ratio', ylabel="density", title='BNC 4M (POS1=%s, POS2=%s)' % (POS1, POS2))
dplot.set_xlim((0, 1))
dplot.axvline(x=0.5, color='r')
plt.show()


# ## Show Examples

# In[13]:


# Top flexible POS1
lemma_count_df[(lemma_count_df['majority_tag'] == POS1) & (lemma_count_df['is_flexible'])].head(10)


# In[14]:


# Examples of inflexible POS1
lemma_count_df[(lemma_count_df['majority_tag'] == POS1) & (~lemma_count_df['is_flexible'])].head(10)


# In[15]:


# Examples of flexible POS2
lemma_count_df[(lemma_count_df['majority_tag'] == POS2) & (lemma_count_df['is_flexible'])].head(10)


# In[16]:


# Examples of inflexible POS2
lemma_count_df[(lemma_count_df['majority_tag'] == POS2) & (~lemma_count_df['is_flexible'])].head(10)


# ## Chi-squared test that POS1 and POS2 are not equally likely to convert

# In[17]:


base_pos1_is_base = lemma_count_df[lemma_count_df.majority_tag == POS1].pos1_count.sum()
base_pos2_is_base = lemma_count_df[lemma_count_df.majority_tag == POS2].pos2_count.sum()
base_pos1_not_base = lemma_count_df[lemma_count_df.majority_tag == POS1].pos2_count.sum()
base_pos2_not_base = lemma_count_df[lemma_count_df.majority_tag == POS2].pos1_count.sum()


# In[18]:


print('Instances of base=%s, pos=%s (no conversion): %d' % (POS1, POS1, base_pos1_is_base))
print('Instances of base=%s, pos=%s (conversion): %d' % (POS1, POS2, base_pos1_not_base))
print('Instances of base=%s, pos=%s (no conversion): %d' % (POS2, POS2, base_pos2_is_base))
print('Instances of base=%s, pos=%s (conversion): %d' % (POS2, POS1, base_pos2_not_base))


# In[19]:


print('Likelihood of %s converting: %0.9f' % (POS1, base_pos1_not_base/base_pos1_is_base))
print('Likelihood of %s converting: %0.9f' % (POS2, base_pos2_not_base/base_pos2_is_base))


# In[20]:


import scipy.stats
pvalue = scipy.stats.chi2_contingency([[base_pos1_is_base, base_pos1_not_base], [base_pos2_is_base, base_pos2_not_base]])[1]
print('p-value from chi-squared test:', pvalue)

