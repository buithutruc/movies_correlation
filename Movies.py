#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Import the packages 
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# Read in the data
df = pd.read_csv(r'/Users/trucb/Documents/DataAnalytist/Python/PythonProjects/Correlation/movies.csv')


# In[4]:


# Look at the data
df.head()


# In[5]:


df


# # EXPLORING DATA

# In[6]:


# Check if there is any missing data
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[7]:


# Check data types of columns
df.dtypes


# In[46]:


# Sort data by gross
df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[24]:


pd.set_option('display.max_rows', None)


# In[27]:


# Drop any duplicates
df['company'].drop_duplicates().sort_values(ascending=False)


# In[28]:


df


# In[44]:





# # Finding Correlation

# In[ ]:


# Check if high budget correlates to high gross
# Check if certain companies correlate to high gross


# ## Check if high budget correlates to high gross

# In[32]:


# Scatter plot with budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget')
plt.show()


# In[31]:


df.head()


# In[34]:


# Plot budget vs gross using seaborn
sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color': 'red'}, line_kws={'color':'green'})


# In[36]:


# Look at correlation
df.corr(method='pearson')


# In[37]:


df.corr(method='kendall')


# In[38]:


df.corr(method='spearman')


# ### Result: High correlation between budget and gross

# In[41]:


correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# ## Check if certain companies correlate to high gross

# In[42]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[47]:


df


# In[48]:


correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[54]:


correlation_matrix = df_numerized.corr()
correlation_pairs = correlation_matrix.unstack()
sorted_correlation_pairs = correlation_pairs.sort_values()
sorted_correlation_pairs


# In[57]:


high_correlation_pairs = sorted_correlation_pairs[(sorted_correlation_pairs) > 0.5]
high_correlation_pairs


# ### Result: Low correlation between companies and gross earnings. However, found high correlation between votes and gross earnings

# ## Conclusion: There is high correlation between budget and gross earnings, votes and gross earnings. Companies seem not to have much influence over a movie's gross earning. 
