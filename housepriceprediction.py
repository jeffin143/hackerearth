from sklearn import linear_model

clf = linear_model.LinearRegression()
F, N = map(int,input().split())  #no of feautres and no of rows
ls =pd.read_csv('housedatset.csv')
un =[]

   
    

ppsqi=[]
for i in range(0,N):
    ppsqi.append(ls[i][-1])
    del ls[i][-1]
    

clf.fit(ls,ppsqi)
for i in clf.predict(un):
    print("%2.f" % i)
