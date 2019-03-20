
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import matplotlib.pyplot as plt
datadigits = load_digits()

X= datadigits.data
Y= datadigits.target


from sklearn.model_selection import StratifiedShuffleSplit
misss=StratifiedShuffleSplit(1,0.3)  #1cortes

misss=StratifiedShuffleSplit(20,0.3)  #20 cortes
fallos=[]
index=0
for train_index, test_index in misss.split(X, Y):

    Xtrain=X[train_index,:]
    Xtest=X[test_index,:]
    Ytrain=Y[train_index]
    Ytest=Y[test_index]

    miKvecinos=KNeighborsClassifier(n_neighbors=3)
    miKvecinos.fit(Xtrain,Ytrain)
    Ypred=miKvecinos.predict(Xtest)
    fallos.append(sum(Ypred!=Ytest))
    index=index+1
        
print ("Num. medio de errores de: " + str(100*np.mean(fallos)/len(Ytest)))
print ("Dev. Std. de errores de: " + str(100*np.std(fallos)/len(Ytest)))


#Grafica con la media de los errores y la desviacion dependiendo 
#del numero de splits que apliquemos y con kvecinos=3 
desv=[]
error=[]
for n in range(1,30):
    misss=StratifiedShuffleSplit(n,0.3)  #20 cortes y una proporción del 30% de la parte de test
    fallos=[] 
    index=0 
    for train_index, test_index in misss.split(X, Y):
        Xtrain=X[train_index,:]
        Xtest=X[test_index,:] 
        Ytrain=Y[train_index] 
        Ytest=Y[test_index] 
        miKvecinos=KNeighborsClassifier(n_neighbors=3) 
        miKvecinos.fit(Xtrain,Ytrain)
        Ypred=miKvecinos.predict(Xtest)         
        fallos.append(sum(Ypred!=Ytest)) 
        index=index+1 

    error.append(100*np.mean(fallos)/len(Ytest))
    desv.append(100*np.std(fallos)/len(Ytest))

    
   

plt.plot(range(1,30),error, 'b-',
         label='Num. medio de errores')
plt.plot(range(1,30), desv, 'r-',
         label='Desv. Std de errores')
plt.legend(loc='upper right')
plt.xlabel('Num. splits')
plt.title('Análisis media/varianza/splits')
plt.ylim(0,4)
plt.style.context('seaborn-whitegrid')
