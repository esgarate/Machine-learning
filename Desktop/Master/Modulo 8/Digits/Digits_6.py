#Comparar los pesos de la distancia en la CV

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.metrics import accuracy_score

datacancer = load_breast_cancer()

data = load_digits()

X= data.data
Y= data.target

#Dividimos el dataset en Train(2/3) y test(1/3)
misss = StratifiedShuffleSplit(1, 0.33)


index = 0
for train_index, test_index in misss.split(X,Y):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    Ytrain = Y[train_index]
    Ytest = Y[test_index]
    index = index +1


#comparo con GridSearch uniform/distance
mi_param_grid={'n_neighbors':[3], 'weights':['uniform','distance']} 
migscv=GridSearchCV(miKvecinos,mi_param_grid,cv=10,verbose=2)
migscv.fit(Xtrain,Ytrain)

#Mejor estimador 
print (migscv.best_estimator_)
#Media y varianza para todos los modelos
print (migscv.grid_scores_)




#Repito el proceso de stratificacion 20 veces
# y saco las accuracys de las 20 iteraciones para peso uniforme y distancia
misss = StratifiedShuffleSplit(20,0.33)
index = 0
ASU = []
ASD = []
for train_index, test_index in misss.split(X,Y):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    Ytrain = Y[train_index]
    Ytest = Y[test_index]
    miKvecinos = KNeighborsClassifier(n_neighbors=3, weights = 'uniform')   
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miKvecinos.predict(Xtest)
    ASU.append(accuracy_score(Ytest, Ypred))
    miKvecinos = KNeighborsClassifier(n_neighbors=3, weights = 'distance')   
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miKvecinos.predict(Xtest)
    accuracy_score(Ytest, Ypred)
    ASD.append(accuracy_score(Ytest, Ypred))
    index = index +1

print (ASU)
print (ASD)



print ("Accuracy de uniform: " + str(ASU))    
print ("Accuracy Media uniform:" + str(np.mean(ASU)))
print ("Dev uniform:" + str(np.std(ASU)))


print ("Accuracy de distance: " + str(ASD))    
print ("Media distance:" + str(np.mean(ASD)))
print ("Dev distance:" + str(np.std(ASD)))
