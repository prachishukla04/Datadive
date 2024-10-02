#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\Prachi\Downloads\supermarket sales Sheet1.csv")
df


# In[35]:


df1=df.copy()   #deep copy to keep one unaltered copy for safety purpose
df1


# In[4]:


df.size


# In[5]:


df.shape


# In[6]:


df.head(10)


# In[7]:


df.tail(10)


# In[8]:


df.info()


# In[10]:


df.describe()


# In[13]:


#Q is your supermarket more popular with males or females

df.groupby('Gender').sum()


# In[21]:


plt.figure(figsize=(10,6))
sns.countplot(x='Gender',data=df,palette='husl')
plt.title("total number of male and female")
plt.grid()


# In[23]:


g=df['Gender'].value_counts()
g


# supermarket is more popular with females however difference is not much

# In[26]:


#Q- what does customer look like and can you also comment on its skewness

import numpy as np
sns.distplot(df['Rating'])
plt.grid()  #there is slight skewness around right side
#majority of rating is densed around 6.5


# In[27]:


#Q- what can you say abt the aggregate sales across the branches

df['Branch'].value_counts()  


# In[28]:


sns.countplot(x=df['Branch'])   #plotting the count plot
plt.title("Count vs Branch")#title of the plot
plt.grid()     


# Sale A is higher than rest of the branches

# In[30]:


#Q4 Which is the most popular payment method used by customers

get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot(x=df['Payment'])
plt.title("count vs method of Payment")
plt.xlabel("Method of Payment")
plt.ylabel("count")


# In[31]:


a=df["Payment"].value_counts()   #payment
a


# Ewallet has been most used

# In[32]:


#Q4b- find branch wise Payment method distribution in all branches?
plt.figure(figsize=(14,6))
plt.style.use('classic')
ax=sns.countplot(x="Payment",hue="Branch",data=df,palette="tab20")
ax.set_title(label="Payment distribution in all branches",fontsize=25)
ax.set_xlabel(xlabel='Payment Method',fontsize=16)
ax.set_ylabel(ylabel='People count',fontsize=16)


# In[33]:


#customer type in different branch
plt.figure(figsize=(14,6))
plt.style.use('classic')
ax=sns.countplot(x="Customer type",hue="Branch",data=df,palette="rocket_r")
ax.set_title(label="Customer type in all branches",fontsize=25)
ax.set_xlabel(xlabel='Branches',fontsize=16)
ax.set_ylabel(ylabel='Customer count',fontsize=16)


# In[5]:


#Q1b- Find rating distribution between branches
plt.figure(figsize=(8,4))
ax=sns.boxplot(x="Branch",y="Rating",data=df,palette="RdYlBu")
ax.set_title("Rating distribution between branches",fontsize=25)
ax.set_xlabel(xlabel="Branches",fontsize=16)
ax.set_ylabel(ylabel="Rating disribution",fontsize=16)
plt.grid()


# In[6]:


#does gross income affect customer ratings?
r=df["Rating"]
r


# In[7]:


sns.set_style('darkgrid')
sns.scatterplot(x=df['Rating'],y=df['gross income'])
plt.title("Gross Income VS Rating")


# In[8]:


#Q2- Which is the most profitable branch?

sns.boxplot(x=df['Branch'],y=df['gross income'])
plt.title("Gross Income Vs Branch")


# In[9]:


#Q3- How is the relationship between Gender and Gross Income?

sns.boxplot(x=df['Gender'],y=df['gross income'])
plt.title("Gross income vs Gender")


# Conclusion - Gross income of female is more than male

# In[11]:


#Q4-Can you see any time trend in gross income

df.groupby(df.index).mean()


# In[12]:


sns.lineplot(x=df.groupby(df.index).mean().index,
            y=df.groupby(df.index).mean()['gross income'])


# Conclusion- Gross income fluctuate alot

# In[15]:


#Q5- which product line helps you generate the most income?

cat=df[["Product line", "gross income"]].groupby(['Product line'], as_index=False).sum().sort_values(by='gross income',ascending=False)
plt.figure(figsize=(20,8))
sns.barplot(x='Product line',y='gross income',data=cat)
plt.title("Gross income vs Product line")


# Conclusion - Gross income is high in food and beverages

# In[18]:


#Q6-What is the spending pattern of both males and females, meaning in which category do they spend more?
plt.figure(figsize=(16,6))
plt.title('Total Monthly transaction by Gender')
sns.countplot(x=df['Product line'], hue= df.Gender)


# Conclusion- female spend more on Fashion accessories and male spend more on Health and Beauty

# In[21]:


#Q7- how many products are bought by customers?
xdata = [1,2,3,4,5,6,7,8,9,10]
plt.figure(figsize = (12,6))
sns.distplot(x=df['Quantity'])


# In[22]:


#Q8- Which day of the week has maximum sales?

df['Date'] = pd.to_datetime(df['Date'])


# In[23]:


df['weekday'] = df['Date'].dt.day_name()


# In[24]:


df.set_index('Date',inplace=True)


# In[25]:


df.head()


# In[27]:


plt.figure(figsize=(8,6))
plt.title('Daily sales by day of the week')
sns.countplot(x=df['weekday'])


# In[30]:


#Q9- Which hour of the day is busiest?

df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = (df['Time']).dt.hour
df['Hour'].unique()


# In[29]:


sns.lineplot(x="Hour", y= 'Quantity',data= df).set_title("Product Sales per Hour")


# Conclusion 14 hours means 2pm is more busiest

# In[32]:


#Q10 - Which product line should your supermarket focus on?
xdata = [0,1,2,3,4,5,6,7,8,9,10]
plt.figure(figsize = (12,6))
sns.barplot(y = df['Product line'],x=df['Rating'])


# Conclusion- Health and lifestyle should be the focus of supermarket

# In[37]:


#Q11- Which city should be chosen for expansion and what products should be focussed on?

plt.figure(figsize=(20,7))
sns.barplot(x=df1['City'],y=df1['gross income'],palette='Set1')
plt.xlabel('City name',fontsize='16')
plt.xticks(fontsize='16')
plt.ylabel('Gross income',fontsize='16')
plt.yticks(fontsize='16')


# Conclusion- Naypyitaw is chosen for expansion and Yangon should be focus on.

# In[39]:


#City

plt.figure(dpi=125)
sns.countplot(y='Product line',hue = "City",data = df)
plt.xlabel('Count')
plt.show()


# In[ ]:




