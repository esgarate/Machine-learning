from sklearn.datasets import load_digits
import numpy as np 
import matplotlib.pyplot as plt
misdatos=load_digits()    #cargo los datos

X=misdatos.data #creo la matriz X 
Y=misdatos.target # vector Y de las etiquetas
arrayx = range(1,1000)
plt.plot(Y, 'r--')
plt.axis([0, 100, 0, 9])
plt.plot(Y)
plt.show()



import matplotlib.pyplot as plt
print('unos de las etiquetas' + str(sum(Y==0)))
Y0=sum(Y==0)
Y1=sum(Y==1)
Y2=sum(Y==2)
Y3=sum(Y==3)
Y4=sum(Y==4)
Y5=sum(Y==5)
Y6=sum(Y==6)
Y7=sum(Y==7)
Y8=sum(Y==8)
Y9=sum(Y==9)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = '0','1', '2', '3', '4','5','6','7','8','9'
sizes = [Y0, Y1, Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9]
explode = (0, 0.1, 0, 0,0,0,0,0,0,0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  
plt.show()



#Accuracy stratificando
miKvecinos=KNeighborsClassifier(n_neighbors=3)
misss = StratifiedShuffleSplit(20,0.33)
misss.split(X,Y)
miKvecinos.fit(X,Y)
index = 0
for train_index, test_index in misss.split(X,Y):
    miKvecinos.fit(X,Y)
    Ypred = miKvecinos.predict(X)
    print (accuracy_score(Ytest, Ypred))
    index = index +1


