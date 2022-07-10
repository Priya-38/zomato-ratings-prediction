import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('final_df.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
X = df.drop('rate', axis=1)
y = df['rate']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=12)
#Preparing Extra Tree Regression
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(X_train,y_train)
y_predict=ET_Model.predict(X_test)


#Using pickle to save our model so that we can use it later

import pickle 
# read model from drive
model=pickle.load(open('model.pkl','rb'))
print(y_predict)
