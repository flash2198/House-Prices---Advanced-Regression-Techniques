import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xg 
from catboost import CatBoostRegressor
from sklearn.svm import SVR


from sklearn.metrics import mean_squared_log_error,mean_absolute_error,mean_squared_error,r2_score,confusion_matrix,classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict


df=pd.read_csv('data/Train.csv')
df_test=pd.read_csv('data/Test.csv')
df_tmp=df.copy()



#<<------------------- handling data ---------------------->>

df_tmp.drop(['PoolQC','MiscFeature'],axis=1,inplace=True)
df_test.drop(['PoolQC','MiscFeature'],axis=1,inplace=True)




for label,content in df_test.items():
	if(pd.api.types.is_string_dtype(content)):
		df_tmp[label]=content.astype('category').cat.as_ordered()

for label,content in df_tmp.items():
	if(pd.api.types.is_string_dtype(content)):
		df_tmp[label]=content.astype('category').cat.as_ordered()




for label,content in df_tmp.items():
	if not(pd.api.types.is_numeric_dtype(content)):
		

		df_tmp[label]=pd.Categorical(content).codes+1

for label,content in df_test.items():
	if not(pd.api.types.is_numeric_dtype(content)):
		

		df_test[label]=pd.Categorical(content).codes+1


df_tmp['LotFrontage'].fillna(np.mean(df_tmp.LotFrontage),inplace=True)
df_test['LotFrontage'].fillna(np.mean(df_test.LotFrontage),inplace=True)

df_tmp['MasVnrArea'].fillna(np.median(df_tmp.MasVnrArea.dropna()),inplace=True)
df_test['MasVnrArea'].fillna(np.median(df_test.MasVnrArea.dropna()),inplace=True)

df_tmp['GarageYrBlt'].fillna(0,inplace=True)
df_test['GarageYrBlt'].fillna(0,inplace=True)

df_test['BsmtFinSF1'].fillna(0,inplace=True)
df_test['BsmtFinSF2'].fillna(0,inplace=True)
df_test['BsmtUnfSF'].fillna(0,inplace=True)
df_test['TotalBsmtSF'].fillna(0,inplace=True)
df_test['BsmtFullBath'].fillna(0,inplace=True)
df_test['BsmtHalfBath'].fillna(0,inplace=True)
df_test['GarageCars'].fillna(2,inplace=True)
df_test['GarageArea'].fillna(0,inplace=True)



for label,content in df_tmp.items():
	if(df_tmp[label].dtype=='float64'):
		df_tmp[label]=df_tmp[label].astype(int)

for label,content in df_test.items():
	if(df_test[label].dtype=='float64'):
		df_test[label]=df_test[label].astype(int)



# splitting the data

x=df_tmp.drop('SalePrice',axis=1)
y=df_tmp['SalePrice']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# predicting function

def predict_me(model):
	print(model)
	y_preds=model.predict(x_test)
	y_preds=np.absolute(y_preds)
	print('model_r^2:',model.score(x_test,y_test))
	print('model_mean_absolute_error:',mean_absolute_error(y_test,y_preds))
	print('model_mean_squared_error:',mean_squared_error(y_test,y_preds))
	print('model_root_mean_squared_error:',np.sqrt(mean_squared_error(y_test,y_preds)))
	print('model_root_mean_squared_log_error:',np.sqrt(mean_squared_log_error(y_test,y_preds)))







# <<<---------------------DecisionTreeRegressor--------------------------->>>

# model=DecisionTreeRegressor(random_state=42)



# ds_grid={"criterion": ["mse", "mae"],
#               "min_samples_split": [10, 20, 40],
#               "max_depth": [2, 6, 8],
#               "min_samples_leaf": [20, 40, 100],
#               "max_leaf_nodes": [5, 20, 100],
#               }

# gs_model=GridSearchCV(model,param_grid=ds_grid,cv=5)

# gs_model.fit(x_train,y_train)

# print(gs_model.best_params_)

# predict_me(gs_model)

# ideal_param={'criterion': 'mae', 'max_depth': 8, 'max_leaf_nodes': 100, 'min_samples_leaf': 20, 'min_samples_split': 10}

# ideal_model=DecisionTreeRegressor(criterion='mae',max_depth=8,max_leaf_nodes=100,min_samples_leaf=20,min_samples_split=10,random_state=42)
# ideal_model.fit(x_train,y_train)
# predict_me(ideal_model)

# y_preds=ideal_model.predict(df_test)

# df_submit=pd.DataFrame()

# df_submit['Id']=df_test['Id']
# df_submit['SalePrice']=y_preds

# df_submit.to_csv('predicted.csv',index=False)









# <<--------------------RandomForestRegressor---------------->>


# ideal_radom={['n_estimators': 50, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 10],
# ['n_estimators': 90, 'min_samples_split': 14, 'min_samples_leaf': 1, 'max_features': 0.5, 'max_depth': 10],n_estimators=90,min_samples_split=16,min_samples_leaf=1,max_features='auto',max_depth=10,random_state=42}

# rf_grid = {"n_estimators": [50,90,100],
#            "max_depth": [None , 10],
#            "min_samples_split":[14,18,6,2],
#            "min_samples_leaf": [1,3,5],
#            "max_features": [0.5, "auto"]}

# rf_model=GridSearchCV(model,param_grid=rf_grid,cv=5)
# rf_model.fit(x_train,y_train)
# print(rf_model.best_params_)
# predict_me(rf_model)


# model=RandomForestRegressor(n_estimators=90,min_samples_split=14,min_samples_leaf=1,max_features='auto',max_depth=10,random_state=42)
# model.fit(x_train,y_train)
# y_preds=model.predict(df_test)
# df_submit=pd.DataFrame()

# df_submit['Id']=df_test['Id']
# df_submit['SalePrice']=y_preds

# df_submit.to_csv('predicted.csv',index=False)





# <<---------------------LinearRegression------------------>>

# model=LinearRegression()
# model.fit(x_train,y_train)


# predict_me(model)





# <<----------------LassoCV--------------->>

# model=LassoCV()

# model.fit(x_train,y_train)

# predict_me(model)







# <<-----------------GradientBoostingRegressor---------------------->>

# model=GradientBoostingRegressor(n_estimators=400,max_depth=5,min_samples_split=2,learning_rate=0.1,loss='ls')

# model.fit(x_train,y_train)

# predict_me(model)

# y_preds=model.predict(df_test)





# gs_grid={'n_estimators':[500,1000,2000],'learning_rate':[0.1],'max_depth':[2,4,5],'min_samples_split':[2,4,6],'subsample':[.5,.75,1]}
# gs_model=GridSearchCV(GradientBoostingRegressor(random_state=42),param_grid=gs_grid,n_jobs=1,cv=5)

# gs_model.fit(x_train,y_train)

# print(gs_model.best_params_)

# predict_me(gs_model)

# ideal_grid={'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 1000, 'random_state': 1, 'subsample': 0.5}



# model_1=GradientBoostingRegressor(n_estimators=1000,max_depth=2,learning_rate=0.1,subsample=0.5)

# model_1.fit(x_train,y_train)

# y_preds=model_1.predict(df_test)


# df_submit=pd.DataFrame()

# df_submit['Id']=df_test['Id']
# df_submit['SalePrice']=y_preds

# df_submit.to_csv('predicted.csv',index=False)



# <<------------------ElasticNetCV---------------->>

# model=ElasticNet()
# model.fit(x_train,y_train)
# predict_me(model)





# <<---------------------RidgeCV--------------------->>

# model=Ridge()

# ridge_params = {'alpha':[10,20,30,50,100,200]}

# rg_model=GridSearchCV(model,param_grid=ridge_params,cv=5)
# rg_model.fit(x_train,y_train)
# print(rg_model.best_params_)
# predict_me(rg_model)




# <<-----------------------SVR-------------------->

# model=SVR(epsilon = 0.01)

# parameters = {'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}

# svm_model=GridSearchCV(model,param_grid=parameters,cv=5)

# svm_model.fit(x_train,y_train)
# print(svm_model.best_params_)
# predict_me(svm_model)



# <<--------------------CatBoostRegressor---------------------->>


# parameters = {'depth'         : [6,8,10],
#                   'learning_rate' : [0.1,0.05],
#                   'iterations'    : [1000]
#                  }
# model_1=CatBoostRegressor(depth=6,learning_rate=0.1)

# model=CatBoostRegressor(iterations=180,depth=6,learning_rate=0.1)
# model_1.fit(x_train,y_train)
# predict_me(model_1)


# y_preds=model_1.predict(df_test)


# df_submit=pd.DataFrame()

# df_submit['Id']=df_test['Id']
# df_submit['SalePrice']=y_preds

# df_submit.to_csv('predicted.csv',index=False)



# <<--------------------xgboost--------------------->>

# model1=xg.XGBRegressor(objective ='reg:squarederror', 
#                   n_estimators = 20, seed = 123)
# model=xg.XGBRegressor()

# model.fit(x_train,y_train)

# predict_me(model)
 
# y_preds=model.predict(df_test)

# df_submit=pd.DataFrame()

# df_submit['Id']=df_test['Id']
# df_submit['SalePrice']=y_preds

# df_submit.to_csv('predicted.csv',index=False)
