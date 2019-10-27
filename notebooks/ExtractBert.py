#!/usr/bin/env python
# coding: utf-8

# # Extract BERT
# 
# Notebook to experiment with libraries to extract parts of BERT embeddings

# In[2]:


import transformers
import torch


# In[ ]:


bert_model = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


# In[10]:


input_ids = bert_tokenizer.encode('My cat is called Xiaonuanhuo and she is warm and fluffy')
input_ids


# In[11]:


len(input_ids)


# In[12]:


bert_embeddings = bert_model(torch.tensor([input_ids]))
bert_embeddings[0]


# In[13]:


# Final hidden layer, one for each token
bert_embeddings[0].shape


# In[14]:


# Pooled layer
bert_embeddings[1].shape


# In[16]:


len(bert_embeddings[2])


# In[19]:


# Nth hidden layer of BERT
N=7
bert_embeddings[2][N].shape

