#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data analysis and interpretations


# In[2]:


#importing required library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing Libraries: Began by importing the necessary libraries to facilitate data manipulation, analysis, and 
# visualization.These libraries included pandas, numpy, seaborn, and matplotlib.pyplot.

# In[3]:


#Looking the data set 
df=pd.read_csv(r"C:\Users\Prachi\Downloads\hrdataset.csv")


# The first step was to locate and load the HR dataset into a pandas DataFrame using the pd.read_csv() function.
# This allowed to access and work with the data efficiently.

# In[4]:


df   #dataset


# In[5]:


df.head(10)   #top 10 data to analyze dataset


# By displaying the first few rows of the DataFrame, obtained a glimpse of the dataset's records and column headers.

# In[6]:


df.tail(10)   #bottom 10 data of datset


# Similarly, inspected the last few rows of the DataFrame to ensure data completeness.

# In[7]:


df.shape


# Used this operation to determine the dimensions of the DataFrame, which includes the number of rows and columns.

# In[8]:


df.describe()


# In[9]:


#filling empty cells of Total_Sales with nan
df=df.replace(r'^\s*$',float(np.NaN), regex=True)


# In[10]:


df.tail()   #to check wheather that values are replaced with nan or not


# In[11]:


df.info()   #info tells type of dataset


# This operation provided essential information about the DataFrame, including the data types of each column and the 
# presence of missing values.

# In[12]:


df['Total_Sales']=df['Total_Sales'].astype('float64')   #changing datatypes of datasets


# In[13]:


df.info()


# In[14]:


missing_values=df.isnull().sum()  #counts the total number of nan value present in dataset
missing_values


# This Series provides a concise summary of missing data in the dataset, which is essential for data quality assessment and 
# determining how to handle missing values during data preprocessing and analysis.

# In[15]:


missing_val_features=[each for each in df.columns if df[each].isnull().sum()>1]
#printing values that have missing values
print(missing_val_features)
#for loop to interate through features and plotting a bar chart

for feature in missing_val_features:
    height=df[feature].isnull().sum()
    plt.bar(feature,height)


# This code is useful for quickly assessing the extent of missing data in your dataset and visualizing it for further 
# analysis and data preprocessing decisions.

# In[16]:


#calculating percentage of missing data
total=df.isnull().sum().sort_values(ascending=False)   #initiating sum of null value of total
percent_1=(df.isnull().sum()/df.isnull().count()*100)   #calcilating total % of null values to percent_1.
percent_2=(round(percent_1,1)).sort_values(ascending=False)   #sorting % of null value in ascending order.
missing_data=pd.concat([total,percent_2],axis=1,keys=['Total','%'])   #joining two column
missing_data.head(3)


# This code helps to assess the extent of missing data in your dataset by providing a summary that includes both the count 
# and percentage of missing values for each column. 

# # imputation

# It is a technique used for replacing missing data with some reasonable value.

# In[17]:


#import simpleImputer from sklearn
from sklearn.impute import KNNImputer
x=df[['Base_pay','openingbalance','Total_Sales']]

#create an object
imputer = KNNImputer(n_neighbors=2)
#fit and transform the values
x=imputer.fit_transform(x)


# This code segment performs KNN-based imputation for the columns 'Base_pay,' 'openingbalance,' and 'Total_Sales' in the 
# DataFrame df. It replaces missing values in these columns with estimated values based on the values of their two nearest 
# neighbors in the dataset. KNN imputation is a technique commonly used when dealing with missing data, especially in 
# scenarios where data points are expected to have similar values when they are close in feature space.

# In[18]:


#converting the array into dataframe
df[['Base_pay','openingbalance','Total_Sales']]=pd.DataFrame(x, columns=['Base_pay','openingbalance','Total_Sales'])


# This ensures that the missing values in these columns are replaced with the imputed values, making the DataFrame complete 
# and ready for further analysis or modeling.

# In[19]:


df.isna().sum()   #now checking again the total null values


# It provides a summary of the number of null values in each column of the DataFrame, helping to verify that the imputation 
# process successfully replaced or reduced the missing data in your dataset.

# In[20]:


#checking the total employee education wise
degree_wise=df.Education.value_counts()  #counting degreewise employee
degree_wise  #degree wise employee


# In summary, this code provides a summary of the distribution of employees based on their education levels. The degree_wise 
# Series shows how many employees belong to each unique education category, helping to understand the educational background 
# of the workforce in your dataset.

# In[21]:


Gender_wise=df.Gender.value_counts()  #counting total number of the employee gender wise.
Gender_wise   #gender wise employee


# So number of male and female employee is almost equal

# In[22]:


#counting how many employees hve been rated by the supervisor
Rating_wise=df.Rating.value_counts()   #counting the total number of ratings employee wise
Rating_wise


# So we can see here that large nuber of employees have not been rated by supervisor.

# In[23]:


#counting how many employees authority to call
call_wise=df.Calls.value_counts()   #counting the total number of employees who have authority to call.
call_wise   


# So large number of employee have authority to call.

# In[24]:


#counting number of people dependent on the person
Dependancies_wise=df.Dependancies.value_counts()   #counting total number of people dependent upon the person.
Dependancies_wise


# This code provides a summary of the distribution of employees based on the number of dependents they have. The 
# Dependancies_wise Series shows how many employees have different numbers of dependents, which can be useful for 
# understanding the family and financial situations of the workforce in the dataset.

# Large number of people are not dependent

# In[25]:


#counting the number of employees working in the company age wise.
Age_wise=df.Age.value_counts()  
Age_wise


# large number of employees working who is in age of 50's and people above 80 are also working.So we can observe that, this 
# company prefers experienced employees thats why large number of people above 50's are working.

# In[26]:


#checking count of salary settlement type
Type_wise=df.Type.value_counts()
Type_wise   #no. of employee settlement type


# We can observe that 2777 people have monthly salary settlement while 1195 people have two year salary settlement and 1028
# people have year wise settlement.

# In[27]:


#checking number of employees Subscribed to billing plans or no
Billing_wise=df.Billing.value_counts()
Billing_wise   #number of employee subscribed to billing plans


# 2956 people have subscribed to billing plans

# In[28]:


#viewing only numerical variables.
df2=df.select_dtypes(include=np.number)
df2.head()


# This code segment extracts and creates a new DataFrame, df2, that includes only the columns with numerical data types from 
# the original DataFrame df. This can be useful when we want to focus on numerical variables for specific data analysis or 
# modeling tasks.

# # Histogram with KDE overlays

# In[29]:


#setting a grey background
sns.set(style='darkgrid')
#creating subplots with numerical variables
fig, axs=plt.subplots(2,2, figsize=(10,10))
#plotting subplots with numerical variables
sns.histplot(data=df,x="Salary",kde=True, color="red",ax=axs[0,0])
sns.histplot(data=df,x="Base_pay",kde=True, color="olive",ax=axs[0,1])
sns.histplot(data=df,x="Bonus",kde=True, color="gold",ax=axs[1,0])
sns.histplot(data=df,x="Age",kde=True, color="black",ax=axs[1,1])
plt.show()


# This code segment creates a 2x2 grid of subplots, each containing a histogram with a KDE overlay for a different numerical 
# variable from the DataFrame df. This allows for a visual exploration of the distribution of these variables.

# In[30]:


fig,axs=plt.subplots(1,2,figsize=(10,5))
sns.histplot(data=df,x="low",kde=True,color="red",ax=axs[0])   # histogram for low column
sns.histplot(data=df,x="Total_Sales",kde=True, color="Navy",ax=axs[1])   # histogram for total_sales column
plt.show()


# This code segment creates two histograms side by side in a single figure, allowing to compare the distributions of the "low"
# and "Total_Sales" variables visually. The histograms include KDE overlays to provide smoother representations of the data 
# distribution.

# # box plot to see distribution
# #from box plot we can see the distribution of the data weather it is normally distributed or not.

# In[31]:


#setting a grey background
sns.set(style="darkgrid")
#creating subplots with 10*10 figure size
fig,axs=plt.subplots(2,2,figsize=(10,10))

#plotting subplots with variables
sns.boxplot(data=df,x="Salary", color="Lime",ax=axs[0,0]) #boxplot to see distribution of salary
sns.boxplot(data=df,x="Base_pay", color="olive",ax=axs[0,1]) #boxplot to see distribution of base pay
sns.boxplot(data=df,x="Bonus", color="gold",ax=axs[1,0])  #boxplot to see distribution of bonus
sns.boxplot(data=df,x="Age", color="pink",ax=axs[1,1])  # boxplot to see distribution of Age
plt.show()


# This code creates a 2x2 grid of subplots, each containing a boxplot for one of the numerical variables. Boxplots are
# useful for visualizing the central tendency and spread of data, as well as identifying potential outliers in the 
# distributions of these variables.

# In[32]:


#setting a grey background
sns.set(style="darkgrid")
#creating subplots with 10*10 figure size
fig,axs=plt.subplots(2,2,figsize=(10,10))

#plotting subplots with variables
sns.boxplot(data=df,x="Unit_Price", color="darkred",ax=axs[0,0]) #boxplot to see outliers of salary
sns.boxplot(data=df,x="Unit_Sales", color="olive",ax=axs[0,1]) #boxplot to see distribution of base pay
sns.boxplot(data=df,x="openingbalance", color="gold",ax=axs[1,0])  #boxplot to see distribution of bonus
sns.boxplot(data=df,x="closingbalance", color="teal",ax=axs[1,1])  # boxplot to see distribution of Age
plt.show()


# This code segment creates a 2x2 grid of subplots, each containing a boxplot for one of the numerical variables. Boxplots are
# useful for visualizing the central tendency, spread, and presence of outliers in the distributions of these variables.

# In[33]:


#setting a grey background
sns.set(style="darkgrid")
#creating subplots with 10*10 figure size
fig,axs=plt.subplots(2,2,figsize=(10,10))

#plotting subplots with variables
sns.boxplot(data=df,x="low", color="red",ax=axs[0,0]) #boxplot to see outliers of low column
sns.boxplot(data=df,x="Total_Sales", color="Navy",ax=axs[0,1]) #boxplot to see distribution of base pay
sns.boxplot(data=df,x="Volume", color="Yellow",ax=axs[1,0])  #boxplot to see distribution of bonus
sns.boxplot(data=df,x="Months", color="skyblue",ax=axs[1,1])  # boxplot to see distribution of Age
plt.show()


# This code segment creates a 2x2 grid of subplots, each containing a boxplot for one of the numerical variables. Boxplots are
# useful for visualizing the central tendency, spread, and presence of outliers in the distributions of these variables.

# # Statistical Approach QQplot to see the distribution

# If the data is normally distributed, the points in the QQ-normal plot lie on a straight diagonal line.

# In[34]:


pip install pingouin


# In[35]:


import pingouin as pg
pg.qqplot(df['Salary'],dist='norm')


# we can see that data are distributed along qq lines hence we can conclude that salary is normally distributed

# In[36]:


pg.qqplot(df['Base_pay'],dist='norm')


# Here also points are along with the qq line, hence it is normally distributed too.

# In[37]:


pg.qqplot(df['Bonus'],dist='norm')


# Bonus is normally distributed here

# In[38]:


#QQ plot for Age
pg.qqplot(df['Age'],dist='norm')


# Age is also normally distributed as points are along with the qq line.

# In[39]:


#QQ plot for Unit_Price
pg.qqplot(df['Unit_Price'],dist='norm')


# Unit_Price is not normally distributed as points are well far from qq-line

# In[40]:


#QQ plot for Unit_Sales
pg.qqplot(df['Unit_Sales'],dist='norm')


# Unit_Sales are not normally distributed.

# In[41]:


#QQ plot for Openingbalance
pg.qqplot(df['openingbalance'],dist='norm')


# Openingbalance is also not normallty distributed as points are not along the QQ-lines.

# In[42]:


#QQ plot for Closing Balance
pg.qqplot(df['closingbalance'],dist='norm')


# not normally distributed as it is not along with the qq-line

# In[43]:


#QQ plot for low
pg.qqplot(df['low'],dist='norm')


# low is also not normally distributed

# # Finding out the correlation between variables using spearman rank correlation

# Pearson correlation assumes that the data we are comparing is normally distributed. When that assumption is not true
# the correlation value is true association. Spearman correlation does not assume that data is from a specific
# distribution, so it is a non parametric correlation measure.
# 
# Thats why we are going to use Spearman rank correlation.

# In[44]:


df3=df2.drop(['Business'],axis=1)   #Here I remove the business column because this variable comes under categorical
df3.head()


# In[45]:


#Spearman rank correlation heatmap

df3.corr(method="spearman")  # selecting the method as a spear man
plt.figure(figsize=(20,8))  #setting the figuresize
heatmap = sns.heatmap(df3.corr(method='spearman').round(3),vmin=-1,vmax=1, annot=True) 
#annot =True means writing the data value in each cell.

font2={'family':'serif','color':'green','size':20}
plt.title("Spearman Rank Correlation",font2)
plt.show()  #displaying heatmap

#jo 80% se jyada correlated h unko drop kar denge
#jo normally distributed nhi h unko drop kar denge


# In[46]:


df3.corr(method='spearman',min_periods=1)   #finding spearman co-relation


# A Spearman correlation of 1 results when the two variables being compared are monotonically related, even if 
# their relationship is not linear.
# Spearman rank correlation coefficient measures the monotonic relation between two variables. Its values range from -1 to +1
# and can be interpreted as:
# 
#    +1:   Perfectly monotonically increasing relationship.
#    
#    +0.8: Strong monotonically increasing relationship.
#    
#    +0.2: Weak monotonically increasing relationship.
#    
#      0:  Non-monotonic relation.
#      
#   -0.2:  Weak monotonically decreasing relationship.
#   
#   -0.8:  Strong monotonically decreasing relationship.
#   
#     -1:  Perfectly monotonically decreasing relationship.
#     
# 
# ---  Here we can see in above table  that spearman correlation between age and salary is 0.2 which means it
#      is weak monotonically increasing relationship.This means if age increases salary will too increase but not that much.
#     
# ---  Spearmen correlation between salary and Total_sales is 0.99.
#      It means it is Perfectly monotonically increasing relationship and salary linearly increases with Total_sales
#      We can depict that employee who will do high Total_sales will get higher salary.
#         
# ---   Similarly we can find relationshio with all variables.

# # The relationship between Categorical variables and Dependent variables.

# In[47]:


#Looking the categorical columns
#dependent variable is Salary here
df.select_dtypes(include=['object']).columns.tolist()


# we will explore the relationships between the categorical columns and our dependent variable, which is the salary of
# employees. This analysis will help us understand how these categorical features impact employee salaries and provide valuable
# insights into our dataset.

# In[48]:


plt.figure(figsize=(12,8))   # size of the plot
sns.boxplot(data=df,x="Gender",y="Salary",color="red")   #boxplot for salary vs gender
plt.title("Salary Vs Gender")   #title of the plot


# There is no major difference between the mean salary of male vs female.

# In[49]:


plt.figure(figsize=(12,8))   # size of the plot
sns.boxplot(data=df,x="Education",y="Salary",color="Green")   #boxplot education wise with salary
plt.title("Salary Vs Education")   #title of the plot


# The box plot analysis reveals clear trends: employees with higher education levels, particularly those with postgraduate
# degrees, tend to command higher mean salaries, while those with high school or intermediate education qualifications
# generally occupy lower-ranking positions in the company, resulting in lower mean salaries for these groups.

# In[50]:


plt.figure(figsize=(12,7))  #figuresize
sns.boxplot(x='Type',y='Salary',data=df,palette='winter')   #Salary vs salary settlement type


# We can see that mean of the all three type of salary settlement is almost equal.

# In[51]:


plt.figure(figsize=(9,7))  #title of the plot
sns.boxplot(x='Business',y='Salary',data=df,palette='winter')  #boxplot Salary vs Business
plt.title("Salary Vs Business")   #title of the plot


# Mean Salary of both the person with or without business is almost same.

# In[52]:


plt.figure(figsize=(9,7))  #size of the plot
sns.boxplot(x='Rating',y='Salary',data=df,palette='winter')  #boxplot rating vs salary
plt.title("Salary Vs Rating")  #title of the salary


# In[53]:


plt.figure(figsize=(9,7))  #size of the plot
sns.boxplot(data=df,x='Dependancies',y='Salary')  #boxplot of dependancies vs salary
plt.title("Salary Vs Dependancies")  #title of the plot


# In[54]:


#Looking to the percentage of corelation
df3.corr(method='spearman')*100


# # scatter plot to see relationship between numerical data

# Creating scatter plots to analyze the relationship between our features and the target variable 'Salary' is a valuable step
# in the data cleaning process. It allows us to identify which features have a discernible impact on 'Salary' and aids in the
# decision to retain or drop certain variables to optimize our model's performance and interpretability.

# In[55]:


#scatter plot vs salary and Base_pay
#distribution of salary on the basis of Base_Pay

plt.figure(figsize=(15,10))
plt.xlabel('Base_pay',fontsize=15)
plt.ylabel('Salary',fontsize=15)
plt.title('Base_pay vs Salary',fontsize=15)
sns.scatterplot(x=df['Base_pay'],y=df['Salary'],color='black')
plt.show()


# By observing the above plot here we see that we have linear relationship between base_pay and Salary 
# so we may keep base_pay in our feature selection

# In[56]:


#distribution of salary on age to know the trend of salary increase according to age or not.
plt.figure(figsize=(15,10))
plt.xlabel('Age',fontsize=15)
plt.ylabel('Salary',fontsize=15)
plt.title('Age vs Salary',fontsize=15)
sns.scatterplot(x=df['Age'],y=df['Salary'],color='black')
plt.show()


# By observing the above plot here we do not see any special patterns in our data 
# there is no relationship between Age and salary and by using
# Spearman correlation we got 0.2 correlation between salary and age thats why we can remove this column in feature selection

# In[57]:


#scatterplot openingbalance vs salary
#distribution of the salary on the basis of openingbalance

plt.figure(figsize=(20,10))
plt.xlabel('openingbalance',fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.title('openingbalance vs Salary',fontsize=20)
sns.scatterplot(x=df['openingbalance'],y=df['Salary'],color='black')
plt.show()


# we can see that there is no clear relationship between openingbalance and salary so we can remove it in feature selection.

# In[58]:


#distribution of salary on volume to know the trend of salary increase according to volume or not.

plt.figure(figsize=(20,10))
plt.xlabel('Volume',fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.title('Volume vs Salary',fontsize=20)
sns.scatterplot(x=df['Volume'],y=df['Salary'],color='black')
plt.show()


# This code segment generates a scatter plot to help you understand how 'Salary' is distributed concerning the 'Volume'
# variable, which can provide insights into whether there is a trend of salary increase with increasing volume.

# In[59]:


#distribution salary on basis of bonus

plt.figure(figsize=(20,10))
plt.xlabel('Bonus',fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.title('Bonus vs Salary',fontsize=20)
sns.scatterplot(x=df['Bonus'],y=df['Salary'],color='black')
plt.show()


# By observing the above plot here we see that we have linear relationship between Bonus and Salary and By using Spearman
# correlation we got 100% correlation between salary and bonus thats why this may be  important variable for us.

# # Data Cleaning and Justification

# The presence of NaN values in the dataset has been addressed by replacing them with the mean values of their respective
# columns. This data preprocessing step helps ensure the accuracy and reliability of the generated graphs and visualizations,
# enabling more meaningful insights and analysis.
# 
# In this data set we will check now for outliers present in the data. Outliers are the extreme value which lies far from
# maximum and minimumof data So sometimes removing outliers increases efficiency of calculation and make the data normally
# distributed.

# # Checking Outliers

# In[60]:


plt.figure(figsize=(9,7))  #size of the figure
figure=df.boxplot(column="Age")   # boxplot of the Age


# We can see the outliers present in Age data

# In[61]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="Salary")   #boxplot of the Salary


# We can observe that outliers are present in the salary.Mean Salary of the employee in this company is approximately 100000.

# In[62]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="Base_pay")  #boxplot of the Salary


# Base_pay has outliers and Mean Base_pay of employees of this company is 40000.

# In[63]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="Bonus")  #boxplot of the Salary


# Bonus has outliers and Mean Bonus is approxmately 5000. 

# In[64]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="Unit_Price")   #boxplot of the Salary


# Unit_price is full of outliers.

# In[65]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="Volume")   #boxplot of the Volume


# Volume has large number of outliers.

# In[66]:


plt.figure(figsize=(9, 7)) #size of the plot
figure=df.boxplot(column="openingbalance") #box plot of the openingbalance


# Openingbalance is full of outliers.

# In[67]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="closingbalance")   #boxplot of the closingbalance


# closebalance has also large number of outliers

# In[68]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="low")   #boxplot of the lowest balance alloted to the person


# It contain large number of outliers.

# In[69]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="Unit_Sales")   #boxplot of the Salary


# unit sales do not have outliers

# In[70]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="Total_Sales")   #boxplot of the Salary


# Total_Sales contain no outliers. 

# In[71]:


plt.figure(figsize=(9,7))  #size of the plot
figure=df.boxplot(column="Months")   #boxplot of the Months.


# It does not contain outliers. Mean duration of the employee working in the company is 28 months approximately.

# # Feature Engineering

# # (Preprocessing the data)

# Now Working on Numerically Highly corelated variables

# From Heatmap I conclude that some of our numerical variables have very high correlation between 
# them which is above 80% colinianity between dependent and other variables.
# 
# Using scatter plot also we saw that relationship between independent and dependent variables 

# In[72]:


#Looking to the percentage of correlation
df3.corr(method='spearman')*100


# # Here we can see that Base_pay, Bonus, Unit_Price, low, Unit_Sales, Total_sales these variables are highly corelated with each other.

# In addressing the multicollinearity issue among these correlated variables, opted for a practical approach: retaining the
# 'bonus' variable, which exhibits a strong correlation with the dependent variable, and eliminating the other highly
# correlated variables from consideration. This choice simplifies the model while preserving the crucial predictive power
# associated with 'bonus.'

# Based on our visualization and analysis, it is evident that the variables 'Gender', 'Business', 'Dependancies', 'Calls',
# 'Type', 'Billing', and 'Rating' do not exhibit a significant impact on our dependent variable. Consequently, we can
# confidently proceed with dropping these columns from our dataset to streamline our analysis and modeling process.

# In[73]:


#Dropping the non required variables
dff=df.drop(columns=['Gender','Business','Dependancies','Calls','Type','Billing','Rating','Base_pay','Unit_Price','Unit_Sales','Total_Sales','low','openingbalance','closingbalance','Volume'])
dff.head()


# In[74]:


#Label encoding to convert categorical data to into numerical data

dff['Education']=dff['Education'].map({'High School or less':0,'Intermediate':1,'Graduation':2,'PG':3})   #assign


# In[75]:


dff.head()  #head of the data set after label encoding


# It is evident that all categorical data in our dataset has been successfully transformed into numerical representations.

# In[76]:


dff.info()


# # Handaling Outliers

# In[77]:


Q1=dff.quantile(0.25).round(3)   #taking lower quantile into Q2
Q3=dff.quantile(0.75).round(3)   #taking upper quantile to Q3
IQR=Q3-Q1  #calculating inter quantile range
print(IQR)  #printing quantile range


# In[78]:


dfout=dff[~((dff<(Q1-1.5*IQR))|(dff>(Q3+1.5*IQR))).any(axis=1)] # removing outliers from whole data set 
dfout.shape


# # sanity check weather outliers are removed or not

# In[79]:


sns.boxplot(data=dfout,x='Bonus')


# we can see there is no outliers in bonus now

# In[80]:


#checking the normal distribution for salary
sns.histplot(data=dff, x="Bonus",kde=True,color="Black")


# We can see that Bonus is normally distributed.

# In[81]:


sns.boxplot(x=dfout['Age'])


# There is no outliers in age.

# In[82]:


sns.histplot(data=dff,x="Age",kde=True,color="Black")


# In[83]:


sns.boxplot(x=dfout['Salary'])


# there is no outliers in salary

# In[84]:


sns.histplot(data=df,x="Salary",kde=True,color="Black")


# In[85]:


sns.boxplot(x=dfout['Months'])


# There is no outliers in month.

# In[86]:


sns.histplot(data=dff,x='Months',kde=True,color='Black')


# # Model Building

# In model building we have to prepare the data set for machine learning algoritham also we have to prepare the data in such 
# a way that it gives better efficiancy. For stability or data set we can drop the highly correlated features in the data set.
# 
# "In the context of machine learning model preparation, ensuring dataset stability and optimizing efficiency often involves
# the identification and elimination of highly correlated features. This step helps reduce redundancy and multicollinearity,
# ultimately improving the performance and interpretability of the model."

# In[87]:


x=dfout.drop(['Salary'],axis=1)   #dropping highly co-related features.
x  #new data set


# This is our dependent Variables

# In[88]:


#dependent variables to y
y=dfout['Salary']
y


# In[89]:


#split x and y into training and testing sets
#mostly devide in 70 & 30% or 75 & 25%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=0)


# In[90]:


x_train


# In[91]:


y_train


# In[92]:


x_test


# In[93]:


y_test


# In[94]:


#data normalization withsklearn
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
#fit scaler on training data
x_train=sc.fit_transform(x_train)

#transform testing dataabsx
x_test=sc.transform(x_test)


# In[95]:


x_train


# In[96]:


x_test


# # Mcahine learning Techniques

# In[97]:


#Importing DecisionTreeRegressor from sklearn.tree library
from sklearn.tree import DecisionTreeRegressor
#calling DecisionTreeRegressor with max_depth as 3 and calling it to dt.
dt=DecisionTreeRegressor(max_depth=3)


# In[98]:


#Fitting x_train, y_train to dt.
dt.fit(x_train,y_train)


# In[99]:


#Score of train Model
print(dt.score(x_train, y_train))
#Score of test model.
print(dt.score(x_test, y_test))


# accuracy is coming 96.97% and 97.11% which is good.

# In[100]:


#By using dt.predict function on x_test and calling it to y_predict to predict a value
y_pred=dt.predict(x_test)


# The code here provided is making use of the predict function to generate predictions based on the input data x_test.

# In[101]:


print(y_pred)  #checking the predicted values


# In[102]:


#importing metrics from sklearn library
from sklearn import metrics


# Metrics gives r2 score to cross verify accuracy of model

# In[103]:


#checking the r2_score to y_predict to predict the accuracy.
#it is good if receive as much as R-square error
#and minimum of RMSE ,MSE, MAE
r_square=metrics.r2_score(y_test,y_pred)
print('R-square error for decision tree regressor is:',round(100*(r_square),3))


# In[104]:


#By using dt.predict function on x_test and calling it to y_predict to predict a value
y_pred=dt.predict(x_test)
#importing metrics from sklearn library
from sklearn import metrics
#Applying r2_score to y_test,y_predict to predict the accuracy and print the accuracy percentage.
r_square=metrics.r2_score(y_test,y_pred)
print('R-square error associated with decision tree regressor is:',round(100*(r_square),2))


# # Rework on different models

# Random forest regressor

# In[105]:


#importing Random forest regressor from sklearn.ensemble and add it to rfr with random_state S and fitting it to x_train
from sklearn.ensemble import RandomForestRegressor
#importing metrices from sklearn library
from sklearn import metrics
rfr =RandomForestRegressor(random_state=5)
rfr.fit(x_train,y_train)


# In[106]:


#by using rfr. predict function x_test and calling it to y_predict to predict a value
y1_pred=rfr.predict(x_test)


# In[107]:


#calculating mean absolute error, mean squared error, root mean squared error, mean absolute percentage error and acc
print('Mean Absolute Error(MAE):',metrics.mean_absolute_error(y_test,y1_pred))
print('Mean Squared Error(MSE):',metrics.mean_absolute_error(y_test,y1_pred))
#while both MAE/MSE are performance measures 
print('Root Mean Squared Error(RMSE):',np.sqrt(metrics.mean_absolute_error(y_test,y1_pred)))
#RMSE is a measure of how will the machine learning model will perform

mape= np.mean(np.abs((y_test-y1_pred)/np.abs(y1_pred)))
#Mean Absolute Percentage Error(MAPE) is a statistical measure to define the accuracy of a machine learning algoritham
#mape=np.mean(np.abs((y_avtual - y_predicted)/y_actual))*100 we use this formula to calculate mape.
mapel=round(mape*100,2)
accuracy1=round(100*(1 - mape),2)

print('Mean Absolute Percentage Error(MAPE):',round(mape*100,2))
print('Accuracy:',round(100*(1-mape),2))


# RandomForestRegressor Accuracy : 99.98

# # Now we are trying to calculate R_score using n_estimators method

# In[108]:


#importing RandomForestRegressor from sklearn.ensemble.
from sklearn.ensemble import RandomForestRegressor
rf =RandomForestRegressor(n_estimators=7)  #here we choose 7 trees only
rf.fit(x_train, y_train)


# In[109]:


#Applying score to x_train, y_train.
rf.score(x_train, y_train)


# In[110]:


#Applying score to x_test, y_test.
rf.score(x_test, y_test)


# # Linear Regresson Model

# In[111]:


#Importing Linear Regression library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[112]:


#Loading Linearregression model
reg_model=LinearRegression()


# In[113]:


reg_model.fit(x_train, y_train)


# In[114]:


#Model evaluation for training set
y_predict=reg_model.predict(x_train)


# In[115]:


rmse=(np.sqrt(mean_squared_error(y_train, y_predict)))
print(rmse)
from sklearn.metrics import r2_score
r2_score(y_train, y_predict)


# In[116]:


#Prediction on test data
y_testpredict = reg_model.predict(x_test)


# In[117]:


#checking the score of model using testing data
r2=r2_score(y_test, y_testpredict)
r2


# It is visible that accuracy is coming 100% which is not a practical scenario hence we can ignore this model while predicting
# accuracy.

# # Cross Validation

# Cross Validation is a resampling method that uses different portions of the data to test and train a model on
# different iterations

# In[118]:


#cross validation for our model
from sklearn.model_selection import ShuffleSplit
model=LinearRegression()
ssplit=ShuffleSplit(n_splits=10,test_size=0.30)
from sklearn.model_selection import cross_val_score
results=cross_val_score(model,x,y,cv=ssplit)
print(results)
print(np.mean(results))


# We can see that for different iteration we are getting accuracy of almost 100% everytime. So 
# linear regression model does not suit this model.

# In[119]:


#cross validation for our model
from sklearn.model_selection import ShuffleSplit
model=DecisionTreeRegressor(max_depth=3)
ssplit=ShuffleSplit(n_splits=10,test_size=0.30)
from sklearn.model_selection import cross_val_score
results=cross_val_score(model,x,y,cv=ssplit)
print(results)
print(np.mean(results))


# For different iteration we are getting good accuracy in decision tree with the mean of 96.98% accuracy.

# In[120]:


#cross validation for our model
from sklearn.model_selection import ShuffleSplit
model=RandomForestRegressor(random_state=5)
ssplit=ShuffleSplit(n_splits=10,test_size=0.30)
from sklearn.model_selection import cross_val_score
results=cross_val_score(model,x,y,cv=ssplit)
print(results)
print(np.mean(results))


# for different iteration we are getting accuracy of almost 99% in the random forest regression.
# 
# The accuracy is varying around 99% for different iteration, which also justify the efficiecy of algoritham

# # Results and Recommendations:

# # Results:

# #Data Analysis and Interpretation:
# 
# 1. The dataset was loaded and explored.
# 2. Both the beginning and end of the dataset were visualized.
# 3. Basic statistical information about the dataset, such as mean and max values, was examined.
# 4. The dataset's shape was determined to be 5000 rows and 20 columns.
# 5. The presence of null values was identified, and steps were taken to address them.
# 6. NaN values were replaced using the KNN imputer for improved data accuracy.
# 7. Anomalies were discovered in the 'Total_Sales' column and corrected.
# 8. Visualizations were created to understand the distribution of various categorical variables.
# 9. Gender distribution showed relatively equal representation of male and female employees.
# 10. Company hiring trends were explored through the 'Months' column.
# 11. The majority of employees were seniors, with an average age of 51.
# 12. Data normality was assessed, revealing the presence of outliers.
# 13. Spearman correlation was calculated, highlighting strong relationships between certain features, particularly 'Salary' and 'Total_Sales.'
# 14. The impact of categorical variables on the dependent variable 'Salary' was analyzed.
# 15. Gender-wise salary distribution showed comparable earnings for both genders.
# 16. Business ownership, authority to make calls, and other factors were examined for their impact on salaries.
# 17. Educational qualifications revealed that postgraduates tended to earn the highest salaries, followed by graduates, while those with lower qualifications had lower salaries.
# 18. Relationships between 'Salary' and numerical features like 'Unit_Sales' and 'Volume' were visualized.
# 19. Data cleaning ensured that all NaN values were replaced with mean values.
# 

# #Data Cleaning and Justification:
# 
# 1. Outliers were identified across multiple numerical features.
# 2. Outliers were removed using the Interquartile Range (IQR) method.
# 3. Box plots were re-plotted to confirm successful outlier removal.
# 4. Categorical features were transformed into numerical representations.
# 5. Feature selection was performed, dropping less impactful features such as 'Calls,' 'Rating,' 'Dependancies,' 'Billing,' 'Type,' 'Business,' 'Gender,' 'Opening Balance,' and 'Closing Balance.'
# 6. Spearman correlation analysis identified and justified the removal of highly correlated features.

# #Feature Engineering:
# 
# 1. Categorical data was converted into numerical format using label encoding.
# 2. Highly correlated features were dropped to improve model stability.
# 3. A threshold of 0.80 was set for feature correlation.
# 4. Features 'Base_pay,' 'Bonus,' 'Total_Sales,' and 'Unit_Sales' were identified as highly correlated and subsequently removed.

# #Model Building:
# 
# 1. Categorical columns ('Calls', 'Rating', 'Dependancies', 'Billing', 'Type', 'Business') were dropped.
# 2. Highly correlated columns were removed to enhance model stability.
# 3. Data was split into training and testing sets (x_train, y_train, x_test, y_test).
# 4. Logarithmic normalization was applied to scale down the target variable 'Salary'.

# #Machine Learning Techniques:
# 
# 1. Decision Tree Regression algorithm was implemented.
# 2. Accuracy achieved was approximately 96.97%.
# 3. Random Forest Regression algorithm was applied, resulting in an accuracy of around 99.98%.
# 4. Linear Regression was attempted, achieving a suspiciously high accuracy of 100%.
# 5. Random Forest Regressor was deemed a more reliable choice.

# #Cross Validation:
# 
# 1. Cross-validation is a resampling method used to assess model performance on different data subsets.
# 2. Decision Tree Regression cross-validation yielded an average accuracy of 96%.
# 3. Random Forest Regression cross-validation resulted in an average accuracy of 99%.
# 4. Linear Regression cross-validation showed an unusually high accuracy of 100%.
# 5. These steps encompass your data exploration, cleaning, feature engineering, model building, and validation processes, providing a comprehensive overview of your data science project.

# # Verdict:

# After thorough exploratory analysis, data cleaning, and model building, it is clear that the Random Forest Regressor
# consistently delivers the highest accuracy compared to other models.

# # Recommendations:

# 1. Prioritize Experienced Candidates: The analysis indicates that employees with more experience contribute significantly
#     to higher sales for the company. Therefore, the company should focus on hiring experienced candidates.
# 
# 2. Leverage Age, Experience, and Education: Employee performance is strongly influenced by age, years of service to the
#     company, and educational qualifications. Strategic hiring decisions should consider these factors, especially favoring candidates with postgraduate education.
# 
# 3. Optimize Salary Considerations: Salary is primarily influenced by age, years of service, and education. HR departments
#     can tailor their hiring strategies based on these factors to attract candidates who can maximize sales, ultimately benefiting the company.
# 
# 4. Align with Total Sales: Company-wide total sales are directly related to employee salaries. HR departments can use this
#     insight to align hiring strategies with sales goals, aiming to hire individuals who can contribute to increased sales
#     revenue.

# In[ ]:




