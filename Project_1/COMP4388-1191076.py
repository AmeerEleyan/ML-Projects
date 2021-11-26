#!/usr/bin/env python
# coding: utf-8

# In[263]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier  


# In[211]:


dataSet = pd.read_csv('Data_set.csv')
dataSet.replace('(^\s+|\s+$)', '', regex=True, inplace=True) # to remove spaces 
dataSet.head()


# In[212]:


# part 1: summary statistics of all attributes
print(dataSet.describe())


# In[213]:


# Part 2: density plot of the entire dataset and split it into two curves (split by Classes, draw the Temperature)
fire = dataSet.query("Classes == 'fire'").Temperature
notFire = dataSet.query("Classes == 'not fire'").Temperature

df_fire=pd.DataFrame(data=fire)
df_notFire=pd.DataFrame(data=notFire)

plt.figure(figsize=(12,8))

plt.plot(df_fire, color='r',linewidth=2.0)
plt.plot(df_notFire, color='b',linewidth=2.0)

plt.xlabel('Days')
plt.ylabel('Temperature')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Temperature Vs. Days split by Classes')

plt.legend(["Fire", "Not Fire"])
plt.show()


# In[214]:


# prt 3: 
columns = ['Temperature','RH','Ws','Rain' ,'FFMC','DMC','DC','ISI','BUI','FWI']
independent_features = dataSet[columns] # Features

mask = np.zeros_like(independent_features.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(16,10))

sns.heatmap(independent_features.corr(),mask=mask ,annot=True, annot_kws={"size":14})
sns.set_style('white')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()


# In[215]:


sns.pairplot(independent_features, corner=True)


# In[268]:


# use this temp in the part 4
temp_data_set = dataSet

df_one = pd.get_dummies(temp_data_set["Classes"])

df_two = pd.concat((temp_data_set, df_one), axis=1)
 
df_two = df_two.drop(["Classes"], axis=1)

df_two = df_two.drop(["fire"], axis=1)
# Fire = 1
# Not fire = 0
temp_for_visualise = df_two.rename(columns={"not fire": "Classes"})


# In[217]:


#part 4
temp_for_visualise[temp_for_visualise.columns[0:14]].corr().iloc[-1:]


# In[218]:



# Part 5
training_data = pd.read_csv('Training_Data.csv')
training_data.replace('(^\s+|\s+$)', '', regex=True, inplace=True) # to remove spaces 

testing_data = pd.read_csv('Testing_Data.csv')
testing_data.replace('(^\s+|\s+$)', '', regex=True, inplace=True) # to remove spaces 


# In[219]:


# part 5.b
linear_model_FWI_One = linear_model.LinearRegression()
linear_model_FWI_One.fit(training_data[['ISI']], training_data.FWI)
print(linear_model_FWI_One.coef_)
print(linear_model_FWI_One.intercept_)


# In[220]:


sns.lmplot(data = training_data, x='ISI', y='FWI', fit_reg=True, line_kws={'color': 'red'})


# In[221]:


# test linear_model_FWI_One
linear_model_FWI_One_pred = linear_model_FWI_One.predict(testing_data[['ISI']])
print(linear_model_FWI_One_pred)


# In[222]:


# Calculate MAE & MSE & RMSE& R_square for linear_model_FWI_One
MAE_linear_model_FWI_One = metrics.mean_absolute_error(testing_data.FWI, linear_model_FWI_One_pred)
MSE_linear_model_FWI_One = metrics.mean_squared_error(testing_data.FWI, linear_model_FWI_One_pred)
RMSE_linear_model_FWI_One = np.sqrt(MSE_linear_model_FWI_One)
R_Square_linear_model_FWI_One = metrics.r2_score(testing_data.FWI,linear_model_FWI_One_pred)


# In[223]:


# Part 5.c
linear_model_FWI_Two = linear_model.LinearRegression()
linear_model_FWI_Two.fit(training_data[['ISI','DMC']], training_data.FWI)
print(linear_model_FWI_Two.coef_)
print(linear_model_FWI_Two.intercept_)


# In[224]:


# test linear_model_FWI_Two
linear_model_FWI_Two_pred = linear_model_FWI_Two.predict(testing_data[['ISI', 'DMC']])
print(linear_model_FWI_Two_pred)


# In[225]:


# Calculate MAE & MSE & RMSE& R_square for linear_model_FWI_Two
MAE_linear_model_FWI_Two = metrics.mean_absolute_error(testing_data.FWI, linear_model_FWI_Two_pred)
MSE_linear_model_FWI_Two = metrics.mean_squared_error(testing_data.FWI, linear_model_FWI_Two_pred)
RMSE_linear_model_FWI_Two = np.sqrt(MSE_linear_model_FWI_Two)
R_Square_linear_model_FWI_Two = metrics.r2_score(testing_data.FWI, linear_model_FWI_Two_pred)


# In[255]:


# part 5.d
linear_model_FWI_Three = linear_model.LinearRegression()
linear_model_FWI_Three.fit(training_data[['ISI','DMC','BUI','DC','FFMC','Temperature']], training_data.FWI)
print(linear_model_FWI_Three.coef_)
print(linear_model_FWI_Three.intercept_)


# In[227]:


# test linear_model_FWI_Three
linear_model_FWI_Three_pred = linear_model_FWI_Three.predict(testing_data[['ISI', 'DMC','BUI','DC','FFMC','Temperature']])
print(linear_model_FWI_Three_pred)


# In[228]:


# Calculate MAE & MSE & RMSE& R_square for linear_model_FWI_Three
MAE_linear_model_FWI_Three = metrics.mean_absolute_error(testing_data.FWI, linear_model_FWI_Three_pred)
MSE_linear_model_FWI_Three = metrics.mean_squared_error(testing_data.FWI, linear_model_FWI_Three_pred)
RMSE_linear_model_FWI_Three = np.sqrt(MSE_linear_model_FWI_Three)
R_Square_linear_model_FWI_Three = metrics.r2_score(testing_data.FWI,linear_model_FWI_Three_pred)


# In[269]:


#part 5.e

first_row = ["Measures type","First Linear Model", "Second Linear Model", "Third Linear Model"]
  
# Add rows 
row_2 = ["MAE",round(MAE_linear_model_FWI_One, 2) , round(MAE_linear_model_FWI_Two,2),  round(MAE_linear_model_FWI_Three, 2)]
row_3 = ["MSE",round(MSE_linear_model_FWI_One,2), round(MSE_linear_model_FWI_Two,2), round(MSE_linear_model_FWI_Three,2)] 
row_4 = ["RMSE", round(RMSE_linear_model_FWI_One,2), round(RMSE_linear_model_FWI_Two,2) ,round(RMSE_linear_model_FWI_Three,2)]
row_5 = ["R-Squared", round(R_Square_linear_model_FWI_One,2) ,round(R_Square_linear_model_FWI_Two,2), round(R_Square_linear_model_FWI_Three,2)]

print(first_row)

print(row_2)

print(row_3)

print(row_4)

print(row_5)


# In[257]:


#part 6
#split dataset in features and target variable
feature_cols = ['day', 'month', 'Temperature','RH','Ws','Rain' ,'FFMC','DMC','DC','ISI','BUI','FWI']

X = training_data[feature_cols] # Features
Y = training_data.Classes # Target variable

logreg = linear_model.LogisticRegression(max_iter=5000)

# fit the model with data
logreg.fit(X,Y)


# In[231]:


log_pred = logreg.predict(testing_data[feature_cols])
log_pred


# In[271]:


log_recall = recall_score(testing_data.Classes, log_pred, average='micro')
log_precision = precision_score(testing_data.Classes, log_pred, average='micro')
log_accuracy = accuracy_score(testing_data.Classes, log_pred)
log_error_rate = 1- log_accuracy
print("Recall of logistic regression model = ",round(log_recall,4))
print("Precision of logistic regression model = ",round(log_precision,4))
print("Accuracy of logistic regression model = ",round(log_accuracy,4))
print("Error_rate of logistic regression model = ",round(log_error_rate,4))


# In[264]:


#part 7
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, Y)


# In[265]:


knn_pred = knn.predict(testing_data[feature_cols])
knn_pred


# In[266]:


knn_recall = recall_score(testing_data.Classes, knn_pred, average='micro')
knn_precision = precision_score(testing_data.Classes, knn_pred, average='micro')
knn_accuracy = f1_score(testing_data.Classes, knn_pred, average='micro')
knn_error_rate = 1 - knn_accuracy
print("Recall of KNN model = ",round(knn_recall,4))
print("Precision of KNN model = ",round(knn_precision,4))
print("Accuracy of KNN model = ",round(knn_accuracy,4))
print("Error_rate of KNN model = ",round(knn_error_rate,4))


# In[ ]:




