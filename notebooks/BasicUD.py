#!/usr/bin/env python
# coding: utf-8

# # Basic UD Part-of-speech Analysis
# 
# Fairly basic analysis involving POS tags on some Universal Dependencies corpora

# In[1]:


from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import conllu

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Read the CoNLL-U file

# In[2]:


#UD_FILE = "../data/zh_gsd-ud-train.conllu"
UD_FILE = "../data/en_ewt-ud-train.conllu"
#UD_FILE = "../data/ja_gsd-ud-train.conllu"

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


print('Noun Flexibility = P(flexible | noun):', noun_flexibility)


# In[13]:


print('Verb Flexibility = P(flexible | verb):', verb_flexibility)


# In[14]:


# Compute ratio of flexible words that are nouns, to compare with Balteiro (2007)
num_flexible = len(lemma_count_df[lemma_count_df['is_flexible']])
num_flexible_nouns = len(lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & lemma_count_df['is_flexible']])
print("Flexibility Asymmetry = P(noun | flexible):", num_flexible_nouns / num_flexible)


# In[15]:


flexible_df = lemma_count_df[lemma_count_df.is_flexible]
dplot = sns.distplot(flexible_df.noun_count / flexible_df.total_count, bins=20)
dplot.set(xlabel='noun ratio', ylabel="density", title=UD_FILE)
dplot.set_xlim((0, 1))
dplot.axvline(x=0.5, color='r')
plt.show()


# ## Show Examples

# In[16]:


# Top flexible nouns
lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (lemma_count_df['is_flexible'])].head(10)


# In[17]:


# Examples of inflexible nouns
lemma_count_df[(lemma_count_df['majority_tag'] == 'NOUN') & (~lemma_count_df['is_flexible'])].head(10)


# In[18]:


# Examples of flexible verbs
lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (lemma_count_df['is_flexible'])].head(10)


# In[19]:


# Examples of inflexible verbs
lemma_count_df[(lemma_count_df['majority_tag'] == 'VERB') & (~lemma_count_df['is_flexible'])].head(10)


# ## Chi-squared test that nouns and verbs are not equally likely to convert

# In[20]:


base_noun_is_base = lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].noun_count.sum()
base_verb_is_base = lemma_count_df[lemma_count_df.majority_tag == 'VERB'].verb_count.sum()
base_noun_not_base = lemma_count_df[lemma_count_df.majority_tag == 'NOUN'].verb_count.sum()
base_verb_not_base = lemma_count_df[lemma_count_df.majority_tag == 'VERB'].noun_count.sum()


# In[21]:


print('Instances of base=N, pos=N (no conversion):', base_noun_is_base)
print('Instances of base=N, pos=V (conversion):', base_noun_not_base)
print('Instances of base=V, pos=V (no conversion):', base_verb_is_base)
print('Instances of base=V, pos=N (conversion):', base_verb_not_base)


# In[22]:


print('Likelihood of noun converting:', base_noun_not_base/base_noun_is_base)
print('Likelihood of verb converting', base_verb_not_base/base_verb_is_base)


# In[23]:


import scipy.stats
pvalue = scipy.stats.chi2_contingency([[base_noun_is_base, base_noun_not_base], [base_verb_is_base, base_verb_not_base]])[1]
print('p-value from chi-squared test:', pvalue)

