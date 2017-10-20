from sklearn import linear_model

clf = linear_model.LinearRegression()
F, N = map(int,input().split())  #no of feautres and no of rows
X_train =pd.read_csv('housedatset.csv')
X_test =[]  #give the dataset for testing the house price

   
    

Y_train=[]
for i in range(0,N):
    Y_train.append(ls[i][-1])
    del X_train[i][-1]
    

clf.fit(X_train,Y_train)
for i in clf.predict(X_test):
    print("%2.f" % i)
