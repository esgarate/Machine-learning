
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
datacancer = load_digits()

X= datacancer.data
Y= datacancer.target



miKvecinos=KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV
mi_param_grid={'n_neighbors':[3,5,7,9,11,13,15],'weights':['uniform','distance']} #número de vecinos: valores a,b,c,d de la rejilla
migscv=GridSearchCV(miKvecinos,mi_param_grid,cv=10,verbose=2)
#verbose hace que salga por pantalla lo que va haciendo
migscv.fit(X,Y)

#migscv.best_estimator_

miMejorKvecinos=migscv.best_estimator_
miMejorKvecinos.fit(X,Y)

#Mejor estimador 
print (migscv.best_estimator_)
#Media y varianza para todos los modelos
print (migscv.grid_scores_)


print (migscv.best_index_)
print (migscv.best_params_)
#cross-validated score of the best_estimator
print (migscv.best_score_)

print (migscv.error_score)

##################################################################
#ploter grafica de scores train/test

import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
datacancer = load_breast_cancer()

X= datacancer.data
Y= datacancer.target

estimator=KNeighborsClassifier()

param_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

train_scores, test_scores = validation_curve(
   estimator, X, Y, param_name="n_neighbors", param_range=param_range,
    cv=2, scoring="accuracy", n_jobs=1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)



plt.title("Validation Curve with GS")
plt.xlabel("Kvecinos")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.xlim(1, 15)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Test score",
             color="navy", lw=lw)

plt.legend(loc="best")
plt.axis([0,15, 0, 1.2])
plt.show()




