
#Valid metrics are ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 
#                   'braycurtis', 'canberra', 'chebyshev', 'correlation', 
#                   'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 
#                   'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 
#                   'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 
#                   'sqeuclidean', 'yule', 'wminkowski'], or 'precomputed', or a 
#                   callable

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import numpy as np 
from sklearn.metrics import accuracy_score
misdatos=load_digits()    #cargo los datos

X=misdatos.data #creo la matriz X 
Y=misdatos.target # vector Y de las etiquetas



#dividimos el Dataset en Xtrain y Xtest
misss = StratifiedShuffleSplit(1, 0.33)

miKvecinos=KNeighborsClassifier()

#aplicamos el metodo GridSearch para hayar la mejor metrica para nuestro modelo con K-vecinos = 5
mi_param_grid={'n_neighbors':[5], 'metric' : ["euclidean", "manhattan", "chebyshev", "minkowski", "l1", "l2", "cityblock", 'hamming', 'jaccard', 'kulsinski'] } 
migscv=GridSearchCV(miKvecinos,mi_param_grid,cv=2,verbose=2)
migscv.fit(X,Y)

print (migscv.best_estimator_)
print (migscv.cv_results_)

miMejorKvecinos=migscv.best_estimator_
miMejorKvecinos.fit(Xtrain,Ytrain)
Ypred = miMejorKvecinos.predict(Xtest)
accuracy_score(Ytest, Ypred)

print (migscv.grid_scores_)
print (migscv.best_score_)




#Repito el proceso de stratificacion 20 veces, entreno con mi mejor con la distancia manhattan
# y saco las accuracys de las 2¶0 iteraciones con Xtest y con Xtrain
misss = StratifiedShuffleSplit(20,0.33)
index = 0
Res = []
Restrain = []
Res1 = []
Restrain1 = []
Res2 = []
Restrain2 = []
Res3 = []
Restrain3 = []
Res4 = []
Restrain4 = []
Res5 = []
Restrain5 = []
Res6 = []
Restrain6 = []
Res7 = []
Restrain7 = []
Res8 = []
Restrain8 = []
Res9 = []
Restrain9 = []
for train_index, test_index in misss.split(X,Y):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    Ytrain = Y[train_index]
    Ytest = Y[test_index]
#Metrica manhattan
    miMejorKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res.append(accuracy_score(Ytest, Ypred))
    Restrain.append(accuracy_score(Ytrain, Ypredtrain)) 
#Metrica euclidean
    miKvecinos=KNeighborsClassifier(n_neighbors=3, weights = 'uniform',metric = 'euclidean')
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res1.append(accuracy_score(Ytest, Ypred))
    Restrain1.append(accuracy_score(Ytrain, Ypredtrain)) 
#Metrica chebyshev
    miKvecinos= KNeighborsClassifier(n_neighbors=3, metric = 'chebyshev')
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res2.append(accuracy_score(Ytest, Ypred))
    Restrain2.append(accuracy_score(Ytrain, Ypredtrain))
#Metrica l1
    miKvecinos= KNeighborsClassifier(n_neighbors=3, metric = 'l1')
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res4.append(accuracy_score(Ytest, Ypred))
    Restrain4.append(accuracy_score(Ytrain, Ypredtrain)) 
#Metrica l2
    miKvecinos= KNeighborsClassifier(n_neighbors=3, metric = 'l2')
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res5.append(accuracy_score(Ytest, Ypred))
    Restrain5.append(accuracy_score(Ytrain, Ypredtrain)) 
#Metrica cityblock
    miKvecinos= KNeighborsClassifier(n_neighbors=3, metric = 'cityblock')
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res6.append(accuracy_score(Ytest, Ypred))
    Restrain6.append(accuracy_score(Ytrain, Ypredtrain)) 
#Metrica hamming
    miKvecinos= KNeighborsClassifier(n_neighbors=3, metric = 'hamming')
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res7.append(accuracy_score(Ytest, Ypred))
    Restrain7.append(accuracy_score(Ytrain, Ypredtrain)) 
#Metrica jaccard
    miKvecinos= KNeighborsClassifier(n_neighbors=3, metric = 'jaccard')
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res8.append(accuracy_score(Ytest, Ypred))
    Restrain8.append(accuracy_score(Ytrain, Ypredtrain)) 
#Metrica kulsinski
    miKvecinos= KNeighborsClassifier(n_neighbors=3, metric = 'kulsinski')
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinos.predict(Xtest)
    Ypredtrain = miMejorKvecinos.predict(Xtrain)
    Res9.append(accuracy_score(Ytest, Ypred))
    Restrain9.append(accuracy_score(Ytrain, Ypredtrain)) 
    index = index +1

#Accuracis prediciendo con Ytest
print (Res)   
#Accuracis prediciendo con Ytrain
print (Restrain)   


print ("Accuracy : " + str(Res))    
print ("Media:" + str(np.mean(Res)))
print ("Dev:" + str(np.std(Res)))

print ("Accuracy de train: " + str(Restrain))    
print ("Media train:" + str(np.mean(Restrain)))
print ("Devtrain:" + str(np.std(Restrain)))

metricas =  ( "manhattan","euclidean", "chebyshev", "l1", "l2", "cityblock", 'hamming', 'jaccard', 'kulsinski')
posicion_x = np.arange(len(metricas))
plt.xticks(posicion_x, metricas,rotation=17)


plt.plot([1, 1], [np.mean(Res),np.mean(Restrain) ])
plt.plot([2, 2], [np.mean(Res1),np.mean(Restrain1) ])
plt.plot([3, 3], [np.mean(Res2),np.mean(Restrain2) ])
plt.plot([4, 4], [np.mean(Res4),np.mean(Restrain4) ])
plt.plot([5, 5], [np.mean(Res5),np.mean(Restrain5) ])
plt.plot([6, 6], [np.mean(Res6),np.mean(Restrain6) ])
plt.plot([7, 7], [np.mean(Res7),np.mean(Restrain7) ]) 
plt.plot([8, 8], [np.mean(Res8),np.mean(Restrain8) ])
plt.plot([9, 9], [np.mean(Res9),np.mean(Restrain9) ]) 
plt.axis([0, 10, 0.98, 1])

plt.show()






