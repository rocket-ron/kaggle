#ughhhh libraries
import csv
import datetime
from sklearn import preprocessing
import numpy as np
from scipy import sparse
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
from sklearn import linear_model
from sklearn import naive_bayes
import time
import pandas as pd

#Load data
train = pd.read_csv('/users/thomasatkins/Documents/MIDS/AML/Rossman/train.csv')
test = pd.read_csv('/users/thomasatkins/Documents/MIDS/AML/Rossman/test.csv')
stores = pd.read_csv('/users/thomasatkins/Documents/MIDS/AML/Rossman/store.csv')

#Join Stores
train = pd.merge(train,stores,on='Store')
test = pd.merge(test,stores,on='Store')

#Remove Sales = 0 Data
train = train[train.Sales>0]

#Convert to log sales
train.Sales = np.log(train.Sales + 1)

#Fill na with 1 for open unknown
train['Open'] = train['Open'].fillna(1)
test['Open'] = test['Open'].fillna(1)

#Add Date Characteristics
date1 = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in train.Date]
date2 = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in test.Date]
train['Day'] = [i.day for i in date1]
train['Month'] = [i.month for i in date1]
train['Year'] = [i.year for i in date1]
test['Day'] = [i.day for i in date2]
test['Month'] = [i.month for i in date2]
test['Year'] = [i.year for i in date2]


#feature engineering - customers per sale and sales per customer by store
def map_cust_per_sale_by_store(dat):
    res = []
    for i in dat:
        res.append([i[0],1.0*float(i[1])/float(i[2]),1.0*float(i[2])/float(i[1])])
    return res


def reduce_by_key(sorted_kv):
    key=1
    values_in_key1=[]
    values_in_key2=[]
    results=[]
    for i in sorted_kv:
        if i[0]==key:
            values_in_key1.append(i[1])
            values_in_key2.append(i[2])
        if i[0]!=key:
            results.append([key,np.median(values_in_key1),np.median(values_in_key2)])
            key = i[0]
            values_in_key1=[i[1]]
            values_in_key2=[i[2]]
    return results



def rmspe(y, yhat):
	return np.sqrt(np.mean((yhat/y-1) ** 2))


def rmspe_xg(yhat, y):
	y = np.expm1(y.get_label())
	yhat = np.expm1(yhat)
	return "rmspe", rmspe(y,yhat)


dat = zip(train.Store,train.Sales,train.Customers)

#mini MapReduce Job
kv = map_cust_per_sale_by_store(dat)
kv.sort(key=lambda x: x[0])
store_cps_spc = reduce_by_key(kv)[1:]


pd_store_cps_spc = pd.DataFrame(store_cps_spc,columns=['Store','ratio1','ratio2'])

train = pd.merge(train,pd_store_cps_spc,on='Store',how='left')
test = pd.merge(test,pd_store_cps_spc,on='Store',how='left')

#fix NAs (all data)
train = train.fillna(0)
test = test.fillna(0)

#label encode
mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
train.StoreType.replace(mappings, inplace=True)
train.Assortment.replace(mappings, inplace=True)
train.StateHoliday.replace(mappings, inplace=True)
mappings2 = {'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3}
train.PromoInterval.replace(mappings2,inplace=True)

test.StoreType.replace(mappings, inplace=True)
test.Assortment.replace(mappings, inplace=True)
test.StateHoliday.replace(mappings, inplace=True)
test.PromoInterval.replace(mappings2,inplace=True)


cat_vars = ['DayOfWeek','Promo','StateHoliday','SchoolHoliday','StoreType','Assortment','CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear','Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval','Day','Month','Year']


num_vars = ['Open','Store','CompetitionDistance','ratio1','ratio2']



X_trn, X_val = train_test_split(train, test_size=0.012, random_state=10)

print 'Training Stage 1 Models'

#train svm
svm1 = LinearSVR(verbose=True)
svm1.fit(X_trn[cat_vars+num_vars],X_trn['Sales'])
svm1_feature = svm1.predict(train[cat_vars+num_vars])
preds = svm1.predict(X_val[cat_vars+num_vars])
print 'svm ',(np.mean(((np.exp(preds)-np.exp(X_val['Sales']))/(np.exp(X_val['Sales'])+1))**2))**0.5


#train xgb
dtrain = xgb.DMatrix(X_trn[cat_vars+num_vars],X_trn['Sales'])
dvalid = xgb.DMatrix(X_val[cat_vars+num_vars],X_val['Sales'])
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

num_boost_round = 50
params1 = {"objective": "reg:linear","booster" : "gbtree",
"eta": 0.5,"max_depth": 2,"subsample": 0.5,"colsample_bytree": 0.4,
"nthread":4,"silent": 1,"seed": 1301}
gbm1 = xgb.train(params1, dtrain, num_boost_round, evals=watchlist,early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)

num_boost_round = 50
params2 = {"objective": "reg:linear","booster" : "gbtree",
"eta": 0.5,"max_depth": 6,"subsample": 0.5,"colsample_bytree": 0.9,
"nthread":4,"silent": 1,"seed": 1301}
gbm2 = xgb.train(params2, dtrain, num_boost_round, evals=watchlist,early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)


xgb1_feature = gbm1.predict(xgb.DMatrix(train[cat_vars+num_vars]))
xgb2_feature = gbm2.predict(xgb.DMatrix(train[cat_vars+num_vars]))


#RandomForestRegressor
X,Y = X_trn[cat_vars+num_vars],X_trn['Sales']
#train model
rfm1 = RandomForestRegressor(n_estimators=77,max_depth=3,n_jobs=-1,verbose=1)
rfm1.fit(X,Y)
rfm1_feature = rfm1.predict(train[cat_vars+num_vars])

rfm2 = RandomForestRegressor(n_estimators=55,max_depth=5,n_jobs=-1,verbose=1)
rfm2.fit(X,Y)
rfm2_feature = rfm2.predict(train[cat_vars+num_vars])

rfm3 = RandomForestRegressor(n_estimators=33,max_depth=10,n_jobs=-1,verbose=1)
rfm3.fit(X,Y)
rfm3_feature = rfm3.predict(train[cat_vars+num_vars])


train['svm1'] = svm1_feature
train['xgb1'] = xgb1_feature
train['xgb2'] = xgb2_feature
train['rfm1'] = rfm1_feature
train['rfm2'] = rfm2_feature
train['rfm3'] = rfm3_feature

#same split
X_trn, X_val = train_test_split(train, test_size=0.012, random_state=10)

print 'Baseline: '

#combine with xgb
dtrain = xgb.DMatrix(X_trn[cat_vars+num_vars],X_trn['Sales'])
dvalid = xgb.DMatrix(X_val[cat_vars+num_vars],X_val['Sales'])
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

num_boost_round = 300
params3 = {"objective": "reg:linear","booster" : "gbtree",
"eta": 0.3,"max_depth": 10,"subsample": 0.95,"colsample_bytree": 0.9,
"nthread":4,"silent": 1,"seed": 1301}
gbm3 = xgb.train(params3, dtrain, num_boost_round, evals=watchlist,early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

preds = gbm3.predict(dvalid)
#score
Y_val = dvalid.get_label()
rmspe1 = (np.mean(((np.exp(preds)-np.exp(Y_val))/(np.exp(Y_val)+1))**2))**0.5
print 'score: ',rmspe1

print 'Stacked Model:'


#combine with xgb
dtrain = xgb.DMatrix(X_trn[cat_vars+num_vars+['xgb1','xgb2','rfm1','rfm2','rfm3','svm1']],X_trn['Sales'])
dvalid = xgb.DMatrix(X_val[cat_vars+num_vars+['xgb1','xgb2','rfm1','rfm2','rfm3','svm1']],X_val['Sales'])
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

num_boost_round = 300
params3 = {"objective": "reg:linear","booster" : "gbtree",
"eta": 0.3,"max_depth": 10,"subsample": 0.95,"colsample_bytree": 0.9,
"nthread":4,"silent": 1,"seed": 1301}
gbm3 = xgb.train(params3, dtrain, num_boost_round, evals=watchlist,early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

preds = gbm3.predict(dvalid)
#score
Y_val = dvalid.get_label()
rmspe1 = (np.mean(((np.exp(preds)-np.exp(Y_val))/(np.exp(Y_val)+1))**2))**0.5
print 'score: ',rmspe1


#baseline model 0.09238 OOS score
#stacked model 0.09829 OOS score

#stacked model didn't improve. this model is overfitting the data since train error
#is 0.074728 for baseline and 0.074210 for stacked model



# dtest = xgb.DMatrix(test[cat_vars+num_vars])
# test_preds = gbm.predict(dtest)

# result = pd.DataFrame({"Id": test["Id"], 'Sales': np.exp(test_preds)})
# result.to_csv("ta_submission_xgb.csv", index=False)

