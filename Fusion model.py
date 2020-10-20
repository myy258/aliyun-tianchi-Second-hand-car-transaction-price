# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:12:16 2020

@author: myy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(123)   

def network(df_train,df_test):
    df_train['bodyType'] = df_train['bodyType'].replace(np.nan, None)
    df_train['model'] = df_train['model'].replace(np.nan, None)
    df_train['fuelType'] = df_train['fuelType'].replace(np.nan, None)
    df_train['gearbox'] = df_train['gearbox'].replace(np.nan, None)
    df_train['notRepairedDamage'] = df_train['notRepairedDamage'].replace('-',None)
    df_train['power'] = df_train['power'].map(lambda x: 600 if x>600 else x)
    df_test['bodyType'] = df_test['bodyType'].replace(np.nan, None)
    df_test['fuelType'] = df_test['fuelType'].replace(np.nan, None)
    df_test['gearbox'] = df_test['gearbox'].replace(np.nan, None)
    df_test['notRepairedDamage'] = df_test['notRepairedDamage'].replace('-', None)
    df_test['power'] = df_test['power'].map(lambda x: 600 if x>600 else x)
       
    tags = ['model','brand','bodyType','fuelType',
            'regionCode','regionCode','regDate','creatDate',
            'kilometer','notRepairedDamage','power',
            'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 
            'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
    
    df_train['power'] = df_train['power'].map(lambda x: 600 if x>600 else x)
    df_test['power'] = df_test['power'].map(lambda x: 600 if x>600 else x)
    
    Standard_scaler = StandardScaler()
    Standard_scaler.fit(df_train[tags].values)
    x = Standard_scaler.transform(df_train[tags].values)
    x_ = Standard_scaler.transform(df_test[tags].values)
    
    y = df_train['price'].values
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    
    model = keras.Sequential([
            keras.layers.Dense(512,activation='relu',input_shape=[26]), 
            keras.layers.Dense(256,activation='relu'), 
            keras.layers.Dense(128,activation='relu'),
            keras.layers.Dense(64,activation='relu'), 
            keras.layers.Dense(1)])
    model.compile(loss='mean_absolute_error',
                    optimizer='Adam')
    
    model.fit(x_train,y_train,batch_size = 256,epochs=200)
    y_ = model.predict(x_)
    data_test_price = pd.DataFrame(y_,columns = ['price2'])
    results = pd.concat([df_test['SaleID'],data_test_price],axis = 1)
    return results

def lgb(df_train,df_test):
    df_train['bodyType'] = df_train['bodyType'].replace(np.nan, -1).astype(int)
    df_train['model'] = df_train['model'].replace(np.nan, -1).astype(int)
    df_train['fuelType'] = df_train['fuelType'].replace(np.nan, -1).astype(int)
    df_train['gearbox'] = df_train['gearbox'].replace(np.nan, -1).astype(int)
    df_train['notRepairedDamage'] = df_train['notRepairedDamage'].replace('-',-1)
  
    df_train['name_count'] = df_train.groupby(['name'])['SaleID'].transform('count')
    df_train['creatDate'] = df_train['creatDate'].astype(str).str[0:4]
    df_train['regDate'] = df_train['regDate'].astype(str).str[0:4]
    df_train['used_year'] = df_train['creatDate'].astype(int) - df_train['regDate'].astype(int)
    df_train['power'] = df_train['power'].map(lambda x: 600 if x>600 else x)
    
    df_train['bodyType_0'] = df_train['bodyType'].apply(lambda x : 1 if x == 0 else 0 )
    df_train['bodyType_1'] = df_train['bodyType'].apply(lambda x : 1 if x == 1 else 0 )
    df_train['bodyType_2'] = df_train['bodyType'].apply(lambda x : 1 if x == 2 else 0 )
    df_train['bodyType_3'] = df_train['bodyType'].apply(lambda x : 1 if x == 3 else 0 )
    df_train['bodyType_4'] = df_train['bodyType'].apply(lambda x : 1 if x == 4 else 0 )
    df_train['bodyType_5'] = df_train['bodyType'].apply(lambda x : 1 if x == 5 else 0 )
    df_train['bodyType_6'] = df_train['bodyType'].apply(lambda x : 1 if x == 6 else 0 )
    df_train['bodyType_7'] = df_train['bodyType'].apply(lambda x : 1 if x == 7 else 0 )
    df_train['bodyType_-1'] = df_train['bodyType'].apply(lambda x : 1 if x == -1 else 0 )   
    df_train['fuelType_0'] = df_train['fuelType'].apply(lambda x : 1 if x == 0 else 0 )
    df_train['fuelType_1'] = df_train['fuelType'].apply(lambda x : 1 if x == 1 else 0 )
    df_train['fuelType_2'] = df_train['fuelType'].apply(lambda x : 1 if x == 2 else 0 )
    df_train['fuelType_3'] = df_train['fuelType'].apply(lambda x : 1 if x == 3 else 0 )
    df_train['fuelType_4'] = df_train['fuelType'].apply(lambda x : 1 if x == 4 else 0 )
    df_train['fuelType_5'] = df_train['fuelType'].apply(lambda x : 1 if x == 5 else 0 )
    df_train['fuelType_6'] = df_train['fuelType'].apply(lambda x : 1 if x == 6 else 0 )
    df_train['fuelType_-1'] = df_train['fuelType'].apply(lambda x : 1 if x == -1 else 0 )
    
    feature_choose0 = ['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType',
           'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode',
           'seller', 'offerType', 'creatDate','price','bodyType_0','bodyType_1',
           'bodyType_2','bodyType_3','bodyType_4','bodyType_5','bodyType_6',
           'bodyType_7','bodyType_-1','fuelType_0','fuelType_1','fuelType_2',
           'fuelType_3','fuelType_4','fuelType_5','fuelType_6','fuelType_-1']
    
    feature_choose0_test = ['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType',
           'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode',
           'seller', 'offerType', 'creatDate','bodyType_0','bodyType_1',
           'bodyType_2','bodyType_3','bodyType_4','bodyType_5','bodyType_6',
           'bodyType_7','bodyType_-1','fuelType_0','fuelType_1','fuelType_2',
           'fuelType_3','fuelType_4','fuelType_5','fuelType_6','fuelType_-1']
    
    feature_choose1 = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 
                      'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 
                      'v_12', 'v_13', 'v_14','used_year','name_count']
    
    feature_choose2 = ['price']
   
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    df_scaler_X = X_scaler.fit_transform(df_train[feature_choose1])
    df_scaler_X1 = pd.DataFrame(df_scaler_X,columns=feature_choose1)
    df_train = pd.concat([df_train[feature_choose0],df_scaler_X1],axis=1)
       
    df_scaler_Y = Y_scaler.fit_transform(df_train[feature_choose2])
    df_scaler_Y1 = pd.DataFrame(df_scaler_Y,columns=['price'])
      
    kk = ['kilometer','power']
    t1 = df_train.groupby(kk[0],as_index=False)[kk[1]].agg(
            {kk[0]+'_'+kk[1]+'_count':'count',kk[0]+'_'+kk[1]+'_max':'max',kk[0]+'_'+kk[1]+'_median':'median',
             kk[0]+'_'+kk[1]+'_min':'min',kk[0]+'_'+kk[1]+'_sum':'sum',kk[0]+'_'+kk[1]+'_std':'std',kk[0]+'_'+kk[1]+'_mean':'mean'})
    df_train = pd.merge(df_train,t1,on=kk[0],how='left') 
    
    train_X = df_train.drop(labels = ['SaleID','price','regDate','creatDate','regionCode','name','offerType','seller'],axis = 1).values
    train_Y = df_scaler_Y1.values 
    x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size = 0.2)
    
    import lightgbm as lgbm
    model_lgbm = lgbm.LGBMRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        boosting_type= 'gbdt',
        objective = 'regression_l1',
        max_depth = -1,
        num_leaves=31,
        min_child_samples = 20,
        feature_fraction = 0.8,
        bagging_freq = 1,
        bagging_fraction = 0.8,
        lambda_l2 = 2,
        random_state=2020,
        metric='mae'
    )
    
    lgbm = model_lgbm.fit(x_train, y_train)
       
    df_out = pd.DataFrame(data=None)
    df_out['SaleID'] = df_test['SaleID']
    df_test['bodyType'] = df_test['bodyType'].replace(np.nan, -1)
    df_test['fuelType'] = df_test['fuelType'].replace(np.nan, -1)
    df_test['gearbox'] = df_test['gearbox'].replace(np.nan, -1)
    df_test['notRepairedDamage'] = df_test['notRepairedDamage'].replace('-', -1)
    df_test['name_count'] = df_test.groupby(['name'])['SaleID'].transform('count')
    df_test['creatDate'] = df_test['creatDate'].astype(str).str[0:4]
    df_test['regDate'] = df_test['regDate'].astype(str).str[0:4]
    df_test['used_year'] = df_test['creatDate'].astype(int) - df_test['regDate'].astype(int)
    df_test['power'] = df_test['power'].map(lambda x: 600 if x>600 else x)
    
    df_test['bodyType_0'] = df_test['bodyType'].apply(lambda x : 1 if x == 0 else 0 )
    df_test['bodyType_1'] = df_test['bodyType'].apply(lambda x : 1 if x == 1 else 0 )
    df_test['bodyType_2'] = df_test['bodyType'].apply(lambda x : 1 if x == 2 else 0 )
    df_test['bodyType_3'] = df_test['bodyType'].apply(lambda x : 1 if x == 3 else 0 )
    df_test['bodyType_4'] = df_test['bodyType'].apply(lambda x : 1 if x == 4 else 0 )
    df_test['bodyType_5'] = df_test['bodyType'].apply(lambda x : 1 if x == 5 else 0 )
    df_test['bodyType_6'] = df_test['bodyType'].apply(lambda x : 1 if x == 6 else 0 )
    df_test['bodyType_7'] = df_test['bodyType'].apply(lambda x : 1 if x == 7 else 0 )
    df_test['bodyType_-1'] = df_test['bodyType'].apply(lambda x : 1 if x == -1 else 0 )
    
    df_test['fuelType_0'] = df_test['fuelType'].apply(lambda x : 1 if x == 0 else 0 )
    df_test['fuelType_1'] = df_test['fuelType'].apply(lambda x : 1 if x == 1 else 0 )
    df_test['fuelType_2'] = df_test['fuelType'].apply(lambda x : 1 if x == 2 else 0 )
    df_test['fuelType_3'] = df_test['fuelType'].apply(lambda x : 1 if x == 3 else 0 )
    df_test['fuelType_4'] = df_test['fuelType'].apply(lambda x : 1 if x == 4 else 0 )
    df_test['fuelType_5'] = df_test['fuelType'].apply(lambda x : 1 if x == 5 else 0 )
    df_test['fuelType_6'] = df_test['fuelType'].apply(lambda x : 1 if x == 6 else 0 )
    df_test['fuelType_-1'] = df_test['fuelType'].apply(lambda x : 1 if x == -1 else 0 ) 
    
    df_scaler_test_X = X_scaler.fit_transform(df_test[feature_choose1])
    df_scaler_test_X1 = pd.DataFrame(df_scaler_test_X,columns=feature_choose1)
    df_test = pd.concat([df_test[feature_choose0_test],df_scaler_test_X1],axis=1)
    
    kk = ['kilometer','power']
    t1 = df_test.groupby(kk[0],as_index=False)[kk[1]].agg(
            {kk[0]+'_'+kk[1]+'_count':'count',kk[0]+'_'+kk[1]+'_max':'max',kk[0]+'_'+kk[1]+'_median':'median',
             kk[0]+'_'+kk[1]+'_min':'min',kk[0]+'_'+kk[1]+'_sum':'sum',kk[0]+'_'+kk[1]+'_std':'std',kk[0]+'_'+kk[1]+'_mean':'mean'})
    df_test = pd.merge(df_test,t1,on=kk[0],how='left')  
    
    df_test = df_test.drop(labels = ['SaleID','regDate','creatDate','regionCode','name','offerType','seller'],axis = 1).values
    test_X = df_test
    
    df_out['price1'] = Y_scaler.inverse_transform(lgbm.predict(test_X)) 
    df_out = df_out[['SaleID','price1']]
    return df_out

if __name__ == "__main__":   
    
    train_file_name = 'used_car_train_20200313.csv'
    test_file_name = 'used_car_testB_20200421.csv'
    
    df_train = pd.read_csv(train_file_name,sep=' ')
    df_test = pd.read_csv(test_file_name,sep=' ')
    
    result1 = lgb(df_train,df_test)
    result2 = network(df_train,df_test)
    
    result = pd.concat([result1,result2['price2']],axis=1)
    result['price'] = result.apply(lambda row: (row['price1'] * 0.3 + row['price2'] * 0.7),axis=1)
    result = result[['SaleID','price']]
    submit_file_z_score = 'test6.csv'
    result.to_csv(submit_file_z_score,encoding='utf8',index=0)

