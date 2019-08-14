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

import src.ud_corpus

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Read the CoNLL-U file

# In[2]:


#UD_FILE = "../data/zh_gsd-ud-train.conllu"
UD_FILE = "../data/en_ewt-ud-train.conllu"
#UD_FILE = "../data/ja_gsd-ud-train.conllu"

ud = src.ud_corpus.UDCorpus(data_file_path=UD_FILE)
ud.data[:3]


# ## POS counts

# In[3]:


pos_counts = defaultdict(int)

for token_list in ud.data:
  for token in token_list:
    pos_tag = token['upostag']
    pos_counts[pos_tag] += 1


# In[4]:


plt.figure(figsize=(12, 6))
plt.bar(pos_counts.keys(), pos_counts.values())


# In[5]:


lemma_count_df = ud.get_per_lemma_stats()
lemma_count_df.sort_values('total_count', ascending=False).head(20)


# ## Distribution of lemmas

# In[6]:


plt.figure(figsize=(15, 5))
lemma_count_df['total_count'].hist(bins=range(0, 60))


# ## Syntax flexibility metrics

# In[7]:


# Only consider lemmas with at least 5 usages
lemma_count_df = lemma_count_df[lemma_count_df['total_count'] >= 5].sort_values('total_count', ascending=False)
noun_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'NOUN'])
verb_lemmas = len(lemma_count_df[lemma_count_df['majority_tag'] == 'VERB'])
noun_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])]) / noun_lemmas
verb_flexibility = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])]) / verb_lemmas


# In[8]:


print('Noun Flexibility = P(flexible | noun):', noun_flexibility)


# In[9]:


print('Verb Flexibility = P(flexible | verb):', verb_flexibility)


# In[10]:


# Compute ratio of flexible words that are nouns, to compare with Balteiro (2007)
num_flexible = len(lemma_count_df[lemma_count_df['is_flexible']])
num_flexible_nouns = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & lemma_count_df['is_flexible']])
print("Flexibility Asymmetry = P(noun | flexible):", num_flexible_nouns / num_flexible)


# In[11]:


flexible_df = lemma_count_df[lemma_count_df.is_flexible]
dplot = sns.distplot(flexible_df.noun_count / flexible_df.total_count, bins=20)
dplot.set(xlabel='noun ratio', ylabel="density", title=UD_FILE)
dplot.set_xlim((0, 1))
dplot.axvline(x=0.5, color='r')
plt.show()


# ## Show Examples

# In[12]:


# Top flexible nouns
lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])].head(10)


# In[13]:


# Examples of inflexible nouns
lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (~lemma_count_df['is_flexible'])].head(10)


# In[14]:


# Examples of flexible verbs
lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])].head(10)


# In[15]:


# Examples of inflexible verbs
lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (~lemma_count_df['is_flexible'])].head(10)


# ## Chi-squared test that nouns and verbs are not equally likely to convert

# In[16]:


base_noun_is_base = lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].noun_count.sum()
base_verb_is_base = lemma_count_df[lemma_count_df.majority_tag == 'VERB'].verb_count.sum()
base_noun_not_base = lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].verb_count.sum()
base_verb_not_base = lemma_count_df[lemma_count_df.majority_tag == 'VERB'].noun_count.sum()


# In[17]:


print('Instances of base=N, pos=N (no conversion):', base_noun_is_base)
print('Instances of base=N, pos=V (conversion):', base_noun_not_base)
print('Instances of base=V, pos=V (no conversion):', base_verb_is_base)
print('Instances of base=V, pos=N (conversion):', base_verb_not_base)


# In[18]:


print('Likelihood of noun converting:', base_noun_not_base/base_noun_is_base)
print('Likelihood of verb converting', base_verb_not_base/base_verb_is_base)


# In[19]:


import scipy.stats
pvalue = scipy.stats.chi2_contingency([[base_noun_is_base, base_noun_not_base], [base_verb_is_base, base_verb_not_base]])[1]
print('p-value from chi-squared test:', pvalue)

