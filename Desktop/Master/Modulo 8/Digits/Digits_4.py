from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from  sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot as plt

data = load_digits()

X= data.data
Y= data.target




############################### LEAVE ONE OUT ####################

#Dividimos el dataset en Train(2/3) y test(1/3)
misss = StratifiedShuffleSplit(1, 0.33)
index=0
for train_index, test_index in misss.split(X, Y):
    Xtrain=X[train_index,:]
    Xtest=X[test_index,:]
    Ytrain=Y[train_index]
    Ytest=Y[test_index]
    index=index+1

#Aplicamos el metodo Grid Search para el elegir el mejor modelo para nuestro data set
#Con CV de leaveOneOut, que es equivalente a hacer una CV con tanatas k-folds como ejemplos
#tiene nuestro dataset, en este caso 569 ejemplos.
miKvecinos=KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV
mi_param_grid={'n_neighbors':[3,4,5,6,7,8,9,10,11], 'weights':['uniform','distance']} 


migscv=GridSearchCV(miKvecinos,mi_param_grid,cv=LeaveOneOut(),verbose=2)
migscv.fit(Xtrain,Ytrain)

#Mejor estimador 
print (migscv.best_estimator_)
#Media y varianza para todos los modelos
print (migscv.grid_scores_)


miMejorKvecinosLOO=migscv.best_estimator_
miMejorKvecinosLOO.fit(Xtrain,Ytrain)
Ypred = miMejorKvecinosLOO.predict(Xtest)
accuracy_score(Ytest, Ypred)


#Repito el proceso de stratificacion 20 veces, entreno con mi mejor modelo de LOO
# y saco las accuracys de las 2¶0 iteraciones con Xtest y con Xtrain
misss = StratifiedShuffleSplit(20,0.33)
index = 0
ResLOO = []
ResLOOtrain = []
for train_index, test_index in misss.split(X,Y):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    Ytrain = Y[train_index]
    Ytest = Y[test_index]
    miMejorKvecinosLOO.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinosLOO.predict(Xtest)
    Ypredtrain = miMejorKvecinosLOO.predict(Xtrain)
    print (accuracy_score(Ytest, Ypred))
    ResLOO.append(accuracy_score(Ytest, Ypred))
    ResLOOtrain.append(accuracy_score(Ytrain, Ypredtrain)) 
    index = index +1

#Accuracis prediciendo con Ytest
print (ResLOO)   
#Accuracis prediciendo con Ytrain
print (ResLOOtrain)   


print ("Accuracy de LOO: " + str(ResLOO))    
print ("Media LOO:" + str(np.mean(ResLOO)))
print ("DevLOO:" + str(np.std(ResLOO)))
print ("Accuracy de LOOtrain: " + str(ResLOOtrain))    
print ("Media LOOtrain:" + str(np.mean(ResLOOtrain)))
print ("DevLOOtrain:" + str(np.std(ResLOOtrain)))

#Grafica con la comparacion de accuracy para Ytest e Ytrain con el mejor modelo
#Y con LOO
plt.plot([3, 3], [np.mean(ResLOO),np.mean(ResLOOtrain) ]) 
plt.axis([0, 10, 0.9, 1])

plt.show()

arrayx= range(1,21)
plt.plot(arrayx,ResLOO, 'r--', label = 'Test')
plt.plot(arrayx,ResLOOtrain, 'g--', label='Train')
plt.axis([0, 20, 0.8, 1.1])
plt.legend(loc='downer right')
plt.xlabel('numero iteracion')
plt.ylabel('Accuracy')
plt.show()


###############  CROSS VALIDATION ######################33

#Dividimos el dataset en Train(2/3) y test(1/3)
misss = StratifiedShuffleSplit(1, 0.33)


#GridSearch Con  10 CV
migscv=GridSearchCV(miKvecinos,mi_param_grid,cv=10,verbose=2)
migscv.fit(Xtrain,Ytrain)


#Mejor estimador 
print (migscv.best_estimator_)
#Media y varianza para todos los modelos
print (migscv.grid_scores_)

miMejorKvecinosCV=migscv.best_estimator_
miMejorKvecinosCV.fit(Xtrain,Ytrain)
Ypred = miMejorKvecinosCV.predict(Xtest)
accuracy_score(Ytest, Ypred)



#Repito el proceso de stratificacion 20 veces, entreno con mi mejor modelo de CV
# y saco las accuracys de las 20 iteraciones con Xtest y con Xtrain
index = 0
ResCV = []
ResCVtrain = []
misss = StratifiedShuffleSplit(20,0.33)

for train_index, test_index in misss.split(X,Y):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    Ytrain = Y[train_index]
    Ytest = Y[test_index]
    miMejorKvecinosCV.fit(Xtrain,Ytrain)
    Ypred = miMejorKvecinosCV.predict(Xtest)
    Ypredtrain = miMejorKvecinosCV.predict(Xtrain)
    print (accuracy_score(Ytest, Ypred))
    print (accuracy_score(Ytrain, Ypredtrain))
    ResCV.append(accuracy_score(Ytest, Ypred))
    ResCVtrain.append(accuracy_score(Ytrain, Ypredtrain))
    index = index +1

print (ResCV)   
#Accuracis prediciendo con Ytrain
print (ResCVtrain) 


#Resultados
print ("Accuracy de CV: " + str(ResCV)) 
print ("Media:" + str(np.mean(ResCV)))
print ("Dev:" + str(np.std(ResCV)))
   
print ("Accuracy de CV train: " + str(ResCVtrain)) 
print ("Media:" + str(np.mean(ResCVtrain)))
print ("Dev:" + str(np.std(ResCVtrain)))



#Grafica con la comparacion de accuracy para Ytest e Ytrain con el mejor modelo
#Y con CV
plt.plot([3, 3], [np.mean(ResCV),np.mean(ResCVtrain) ]) 
plt.axis([0, 10, 0.9, 1])

plt.show()

arrayx= range(1,21)
plt.plot(arrayx,ResCV, 'r--', label = 'Test')
plt.plot( arrayx,ResCVtrain, 'g--', label = 'Train')
plt.axis([0, 20, 0.8, 1.1])
plt.legend(loc='downer right')

plt.xlabel('numero iteracion')
plt.ylabel('Accuracy')
plt.show()




#Pintar grafica con las curvas de la accuracy para train y para test 
import matplotlib.pyplot as plt


plt.plot([11, 11], [np.mean(ResCV),np.mean(ResCVtrain)] ,label='CV' )
plt.plot([3, 3], [np.mean(ResLOO),np.mean(ResLOOtrain) ], label = 'LeaveOneOut') 
 
plt.axis([0, 12, 0.9, 1])
plt.legend(loc='downer right')
plt.show()







#Mejor estimador 
print (migscv.best_estimator_)
#Media y varianza para todos los modelos
print (migscv.grid_scores_)


print (migscv.best_index_)
print (migscv.best_params_)
#cross-validated score of the best_estimator
print (migscv.best_score_)

print (migscv.error_score)
print (migscv.cv_results_)

print (migscv.grid_scores_)
print (miMejorKvecinos)

