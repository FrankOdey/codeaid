'''import the important modules for allround analysis'''
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
recycle=pd.DataFrame()
def merge(train,test):
	'''
	merge two datasets together for easier data manipulation on both datasets
	'''
	train['source']='train'
	test['source']='test'
	data=pd.concat([train,test],ignore_index=True)
	return data
def split(data):
	'''
	split two datasets that were previously merged together into train and test
	'''
	train=data[data['source']=='train']
	test=data[data['source']=='test']
	del train['source']
	del test['source']
	test.dropna(inplace=True,axis=1)
	return train,test
def get_y_x(train,target):
	'''get the independent and dependent variables'''
	y=train[target]
	x=train.drop(columns=[target])
	return x,y

def get_validation_set(train,n):
	''' get a validation set'''
	num=len(train)-n
	return train[:num].copy(), train[num:].copy()

def viewmissing(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
def hour_split(data,hour):
	data['hours']=0
	data.loc[data[hour]<12,'hours']=1
	data.loc[(data[hour]>=12)&(data[hour]<16),'hours']=2
	data.loc[(data[hour]>=16)&(data[hour]<20),'hours']=3
	data.loc[(data[hour]>=19)&(data[hour]<=24),'hours']=4

def date_split(data,date):
	a=pd.to_datetime(data[date])
	data['weekday']=a.dt.dayofweek
	data['year']=a.dt.dayofyear
	data['day']=a.dt.day
	data['quarter']=a.dt.quarter
	data['is_weekend']=0
	data['is_monthend']=0
	data['is_monthstart']=0
	data.loc[(data['day']>=26),'is_monthend']=1
	data.loc[(data['day']<=5),'is_monthstart']=1
	data.loc[(data['weekday']>=4),'is_weekend']=1
def drop(data,column):
	recycle[column]=data[column].copy()
	data.drop(columns=column,inplace=True)
def pick(data,column):
	data[column]=recycle[column]
def catcode(data,column):
    data[column]=data[column].astype('category')
    data[column]=data[column].cat.codes
def scatter_features(data,model,c='r'):
    feat=pd.DataFrame(data.columns,model.feature_importances_)
    feat.reset_index(inplace=True)
    feat['index']=(feat['index']*1000).astype('int64')
    plt.scatter(feat['index'],feat[0],c=c)
def print_score(m):
    res = [mae(m.predict(X_train), y_train), mae(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print(res)
def show_functions():
	print('merge(train,test):'+' '+'This function merges two datasets(train and test)')
	print('split(data):'+' '+'This function splits a dataset that was merged using the merge function')
	print('get_y_x(train,target):'+' '+'This function gets the independent variables and the dependent variable into two containers')
	print('viewmissing(df):'+' '+'view the missing columns by percentage')
