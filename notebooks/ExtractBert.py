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


def convert_to_bert_input(sentences):
  def pad_to_length(tokens, desired_len):
    return tokens + (['[PAD]'] * (desired_len - len(tokens)))
  bert_tokens = [bert_tokenizer.tokenize(sentence) for sentence in sentences]
  max_len = max([len(tokens) for tokens in bert_tokens])
  padded_tokens = [pad_to_length(tokens, max_len) for tokens in bert_tokens]
  padded_ids = [bert_tokenizer.encode(tokens) for tokens in padded_tokens]
  attn_mask = [[1 if token != '[PAD]' else 0 for token in tokens] for tokens in padded_tokens]
  return padded_tokens, padded_ids, attn_mask


# In[4]:


STR1 = "My cat is called Xiaonuanhuo and she is warm and fluffy"
STR2 = "this is a much shorter sentence"
padded_tokens, padded_ids, attn_mask = convert_to_bert_input([STR1, STR2])
print(padded_tokens)
print(padded_ids)
print(attn_mask)


# In[5]:


bert_embeddings = bert_model(torch.tensor(padded_ids), attention_mask=torch.tensor(attn_mask))
bert_embeddings[0]


# In[6]:


bert_embeddings[0].shape


# In[7]:


# Final hidden layer, one for each token
bert_embeddings[0].shape


# In[8]:


# Pooled layer
bert_embeddings[1].shape


# In[9]:


len(bert_embeddings[2])


# In[10]:


# Nth hidden layer of BERT
layer = 7
bert_embeddings[2][layer].shape

