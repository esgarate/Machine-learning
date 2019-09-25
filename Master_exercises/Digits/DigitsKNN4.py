from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


import numpy as np 
misdatos=load_digits()    

X=misdatos.data
Y=misdatos.target
miPCA=PCA(n_components=2)
#eliminamos columnas de features para poder pintar proyeccción 
#por componentes principales pongo 2 que es igual que el num columnas transformadas
X_PCA=miPCA.fit_transform(X)
#has proyectado todo manteniendo la máxima varianza 
#eliminas la varianza cero porque no te aporta nada nuevo

plt. scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=Y)
plt.show()
#Vamos a pintar las etiquetas con s=100, c=Y



#grafica covarianza 
from sklearn.decomposition import PCA
covar_matrix = PCA(n_components = 30) #we have 30 features
covar_matrix.fit(X)
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print(var)
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
#plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')


plt.plot(var)
