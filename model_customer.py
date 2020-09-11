import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
import pandas as pd

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


del df['customerID']
#print(df)

print(df.shape)
#to change string to number 
#to chack the if have other string columns
df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()]

print(df.info())

df.iloc[488]['TotalCharges']
df = df[df.TotalCharges != ' ']
print(df.info())
print(df.shape)

#to convert the string data to numeric values
df.TotalCharges = pd.to_numeric(df.TotalCharges)
df.TotalCharges.dtypes

# to check how many customer he was deal with company
customer_no = df[df.Churn == 'No'].tenure
customer_yes = df[df.Churn == 'Yes'].tenure
print(f'This customer no cluses: {customer_no}')
print(f'This customer yes cluses: {customer_yes}')

# how to compere values with anther columns


# this how hist data and color
#plt.hist([customer_no,customer_yes], color=['green','red'], label=['churn=No,','churn=Yes'])
#plt.legend()
#plt.show()


print(df['Churn'].value_counts())


# ro see envery columns unique columns in data set
def print_unique_col_values(df):
    for column in df:
       if df[column].dtypes == 'object':
          print(f'{column} : {df[column].unique()}')
    
# if you want change sting in values or intager    
df.replace('No internel service','No', inplace=True)
df.replace('No phone service','No', inplace=True)    

# to all unique columnd thouth functions
print(f'Unique columns: {print_unique_col_values(df)}')


# to use change categrical values to number yes: no:0
df['Partner'].replace({'Yes':1, 'No':0}, inplace=True)
df['Dependents'].replace({'Yes':1, 'No':0}, inplace=True)
df['PhoneService'].replace({'Yes':1, 'No':0}, inplace=True)
df['MultipleLines'].replace({'Yes':1, 'No':0}, inplace=True)
#df['MultipleLines'].replace({'Yes':1, 'No':0}, inplace=True)
df['gender'].replace({'Female':1, 'Male':0}, inplace=True)
df['PaperlessBilling'].replace({'Yes':1, 'No':0}, inplace=True)
df['Churn'].replace({'Yes':1, 'No':0}, inplace=True)






# demmies funcation to change string to number as well
#pd.get_dummies(data=df,columns=['InternalService'])

# ot dummies to group of columns
df_dmmies = pd.get_dummies(data=df, columns=['InternetService','Contract','PaymentMethod','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])

#this how shhaow sample in data or dummies
print(df.sample(2))

print(df_dmmies.sample(4))
print(df_dmmies.shape)
print(df_dmmies.info())
print(df.head())

def print_unique_col_values(x):
    for column in x:
       if x[column].dtypes == 'object':
          print(f'{column} : {x[column].unique()}')

print_unique_col_values(df_dmmies)


print(df_dmmies)

#scaler some of columns for the values in columns more then one will scale to 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
col_to_scaler = ['tenure','TotalCharges','MonthlyCharges']
df_dmmies[col_to_scaler] =scaler.fit_transform(df_dmmies[col_to_scaler])

print(df_dmmies)

for i in df_dmmies:
    print(f'{i}:  {df_dmmies[i].unique()}')


print(df_dmmies.shape)


x, y = df_dmmies.drop('Churn', axis=1).values, df_dmmies['Churn'].values

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=0.30)


#model 
model = keras.Sequential([
       keras.layers.Dense(38, input_shape=(38,),activation='relu'),
       keras.layers.Dense(1, activation='sigmoid'),
    
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(xtrain, ytrain, epochs=1000)

print(f'Evalualte Model:{model.evaluate(xtest, ytest)}')
      
pred = model.predict(xtest)
print(pred[:5])

y_pred = []
for elment in pred:
    if elment > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)  
        
        
from sklearn.metrics import accuracy_score

print(accuracy_score(ytest, y_pred))


from seaborn as sns
from sklearn.metrics import confusion_matrix

cm = tf.math.confusion_matrix(labels=ytest), predictions=y_pred)
plt.figure(figsize=(10,7))
plt.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()        
        