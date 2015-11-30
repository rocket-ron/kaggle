import csv
import datetime
from sklearn import preprocessing
import numpy as np
from scipy import sparse
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import naive_bayes
import time

def load(file):
    #open file
    inF = open(file,'rb')
    rdr = csv.reader(inF,delimiter=',')

    #read column names
    cols = rdr.next()

    #extract data
    dat = []
    for l in rdr:
        dat.append(l)
    return dat

dat = load('train.csv')
test = load('test.csv')

#extract target var
Y = np.array([float(i[3]) for i in dat])

#drop sales,customers variables as not in test set
dat = [dat[i][:3] + dat[i][5:] for i in range(len(dat))]

#drop id from test set
test = [test[i][1:] for i in range(len(test))]

#convert dates to dt
for i in range(len(dat)):
    d = datetime.datetime.strptime(dat[i][2],'%Y-%m-%d')
    dat[i] = dat[i][:2] + dat[i][3:] + [str(d.year),str(d.month),str(d.day)]

for i in range(len(test)):
    d = datetime.datetime.strptime(test[i][2],'%Y-%m-%d')
    test[i] = test[i][:2] + test[i][3:] + [str(d.year),str(d.month),str(d.day)]

#encode string to int
le1,le2,le3,le4,le5,le6,le7,le8,le9 = preprocessing.LabelEncoder(),preprocessing.LabelEncoder(),preprocessing.LabelEncoder(),preprocessing.LabelEncoder(),preprocessing.LabelEncoder(),preprocessing.LabelEncoder(),preprocessing.LabelEncoder(),preprocessing.LabelEncoder(),preprocessing.LabelEncoder()
le1.fit([i[0] for i in dat])
le2.fit([i[1] for i in dat])
le3.fit([i[2] for i in dat])
le4.fit([i[3] for i in dat])
le5.fit([i[4] for i in dat])
le6.fit([i[5] for i in dat])
le7.fit([i[6] for i in dat])
le8.fit([i[7] for i in dat])
le9.fit([i[8] for i in dat])

#transform train data
X_cat1 = le1.transform([i[0] for i in dat])
X_cat2 = le2.transform([i[1] for i in dat])
X_cat3 = le3.transform([i[2] for i in dat])
X_cat4 = le4.transform([i[3] for i in dat])
X_cat5 = le5.transform([i[4] for i in dat])
X_cat6 = le6.transform([i[5] for i in dat])
X_cat7 = le7.transform([i[6] for i in dat])
X_cat8 = le8.transform([i[7] for i in dat])
X_cat9 = le9.transform([i[8] for i in dat])
X_cat = [[X_cat1[i],X_cat2[i],X_cat3[i],X_cat4[i],X_cat5[i],X_cat6[i],X_cat7[i],X_cat8[i],X_cat9[i]] for i in range(len(X_cat1))]

#fix up test data i[2] contains the 'open' variable which in the test set has some unknown values
#here we assume if open is unknown then the shop is open
for i in test:
    if i[2]=='':
        i[2]='1'

#transform test data
test_cat1 = le1.transform([i[0] for i in test])
test_cat2 = le2.transform([i[1] for i in test])
test_cat3 = le3.transform([i[2] for i in test])
test_cat4 = le4.transform([i[3] for i in test])
test_cat5 = le5.transform([i[4] for i in test])
test_cat6 = le6.transform([i[5] for i in test])
test_cat7 = le7.transform([i[6] for i in test])
test_cat8 = le8.transform([i[7] for i in test])
test_cat9 = le9.transform([i[8] for i in test])
test_cat = [[test_cat1[i],test_cat2[i],test_cat3[i],test_cat4[i],test_cat5[i],test_cat6[i],test_cat7[i],test_cat8[i],test_cat9[i]] for i in range(len(test_cat1))]

#create dummy vars
enc = preprocessing.OneHotEncoder(sparse=True)
enc.fit(X_cat)
X = enc.transform(X_cat)
test = enc.transform(test_cat)

#don't need stores with 0 sales
X = X[Y>0]
Y = Y[Y>0]

#log transform sales
Y = np.log(Y)

#Do some cross val testing
kf = KFold(np.shape(X)[0], n_folds=5)
i=0
rmspe=[]
t1=time.time()
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    #Scale Data
    #Scale X
    scaler = preprocessing.StandardScaler(with_mean=False)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #train model
    mod = RandomForestRegressor(n_estimators=20,n_jobs=-1) #very slow, especially on all data
    #mod = GradientBoostingRegressor(n_estimators=20)
    #mod = linear_model.SGDRegressor()#~500 error requires scaling
    #mod = naive_bayes.GaussianNB() #doesn't work with sparse
    mod.fit(X_train_scaled[:30000],Y_train[:30000])
    #make predictions
    preds = np.exp(mod.predict(X_test_scaled))
    Y_test = np.exp(Y_test)
    #score
    rmspe.append((np.mean(((preds-Y_test)/(Y_test+1))**2))**0.5)
    print i+1,rmspe[i]
    i=i+1
print 'Time =',int(time.time()-t1),'s'
print 'RMSPE avg =',np.mean(rmspe)

#apply model to test data and export results
test_scaled = scaler.transform(test)
test_preds = np.exp( mod.predict(test_scaled) )

outF = open('sub3.csv','wb')
fwriter = csv.writer(outF,delimiter=',')
fwriter.writerow(['Id','Sales'])
for i in range(len(test_preds)):
    fwriter.writerow([i+1,int(test_preds[i])])
outF.close()