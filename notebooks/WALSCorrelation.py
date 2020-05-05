#!/usr/bin/env python
# coding: utf-8

# # WALS Correlation

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


values_df = pd.read_csv('../data/wals-2020/cldf/values.csv')
parameters_df = pd.read_csv('../data/wals-2020/cldf/parameters.csv')
codes_df = pd.read_csv('../data/wals-2020/cldf/codes.csv')
language_names_df = pd.read_csv('../data/wals-2020/cldf/language_names.csv')


# ## Denormalize values table

# In[3]:


language_names_df = language_names_df.groupby('Language_ID').first()
df = pd.merge(values_df, language_names_df, on='Language_ID')
df = df[['Language_ID', 'Name', 'Parameter_ID', 'Value']].rename(columns={'Name': 'Language_Name', 'Value': 'Parameter_Value'})
df = pd.merge(df, parameters_df, left_on='Parameter_ID', right_on='ID')
df = df[['Language_ID', 'Language_Name', 'Parameter_ID', 'Name', 'Area', 'Parameter_Value']].rename(columns={'Name': 'Parameter_Name'})
df = pd.merge(df, codes_df, left_on=('Parameter_ID', 'Parameter_Value'), right_on=('Parameter_ID', 'Number'))
df = df[['Language_ID', 'Language_Name', 'Area', 'Parameter_Name', 'Name', 'Description']].rename(columns={'Name': 'Value'})
df


# ## Compare two languages on differences

# In[4]:


lang1_df = df[df.Language_ID == 'slo']
lang2_df = df[df.Language_ID == 'pol']
both_df = pd.merge(lang1_df, lang2_df, on='Parameter_Name')
both_df = both_df[['Area_x', 'Parameter_Name', 'Value_x', 'Value_y']]
both_df = both_df.rename(columns={'Value_x': 'Lang1', 'Value_y': 'Lang2'})
both_df[both_df.Lang1 != both_df.Lang2]

