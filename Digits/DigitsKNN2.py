import sklearn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
datadigits = load_digits()

X= datadigits.data
Y= datadigits.target

#Dividimos el dataset en Train(2/3) y test(1/3)
misss = StratifiedShuffleSplit(1, 0.33)
index=0
for train_index, test_index in misss.split(X, Y):
    Xtrain=X[train_index,:]
    Xtest=X[test_index,:]
    Ytrain=Y[train_index]
    Ytest=Y[test_index]
    index=index+1



#elegimos como estimador kvecinos  con 3 vecinos
miKvecinos = KNeighborsClassifier(n_neighbors=3 )   

#hacemos cros validacion con el estimador elegido y 10 kfolds
#El metodo Cros_val_score ya entrena el modelo

micvs = cross_val_score(miKvecinos, Xtrain, Ytrain,cv=10)
print (micvs) 
 
Ypred = cross_val_predict(miKvecinos, Xtest, Ytest,cv=10)
accuracy_score(Ytest, Ypred)

print (micvs)
print ("Mean:" + str(np.mean(micvs)))
print ("Std:" + str(np.std(micvs)))



#Repito el proceso 20 veces estratificando el dataset para obtener la curva
misss = StratifiedShuffleSplit(20,0.33)
index = 0
Res = []
Restrain = []
Ytrain = []
Ytest = []
for train_index, test_index in misss.split(X,Y):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    Ytrain = Y[train_index]
    Ytest = Y[test_index]
    Restrain = cross_val_score(miKvecinos, Xtrain, Ytrain,cv=20)
    Restest = cross_val_score(miKvecinos, Xtest, Ytest,cv=20)
    index = index +1



#Grafica de las accuracys para Test y Train

arrayx= range(1,21)

plt.plot(arrayx,Restrain, 'r--',label='Accuracy Train')
plt.plot(arrayx,Restest, 'g--',label='Accuracy Test')
plt.axis([0, 20, 0.5, 1.2])
plt.legend(loc='up right')

plt.xlabel('numero CV')
plt.ylabel('Accuracy')
plt.show()


