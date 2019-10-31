#!/usr/bin/env python
# coding: utf-8

# # Extract BERT
# 
# Notebook to experiment with libraries to extract parts of BERT embeddings

# In[1]:


import transformers
import torch


# In[ ]:


bert_model = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


# In[3]:


STR = "My cat is called Xiaonuanhuo and she is warm and fluffy"
bert_tokenizer.tokenize(STR)


# In[4]:


input_ids = bert_tokenizer.encode(STR)
input_ids


# In[5]:


len(input_ids)


# In[6]:


input_ids2 = bert_tokenizer.encode("this is a much shorter sentence")
len(input_ids2)


# In[7]:


bert_embeddings = bert_model(torch.tensor([input_ids]))
#bert_embeddings = bert_model(torch.tensor([input_ids, input_ids2]))
bert_embeddings[0]


# In[8]:


bert_embeddings[0].shape


# In[9]:


# Final hidden layer, one for each token
bert_embeddings[0].shape


# In[10]:


# Pooled layer
bert_embeddings[1].shape


# In[11]:


len(bert_embeddings[2])


# In[12]:


# Nth hidden layer of BERT
layer = 7
bert_embeddings[2][layer].shape

