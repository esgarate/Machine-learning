# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:23:31 2018

@author: usuario
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier 


datadigits = load_digits()

X= datadigits.data
Y= datadigits.target


#2) Describe el dataset en dimensiones como en número de características, número de categorías y número
#de samples por categoría utilizando Python

print ('Description: ' + datadigits.DESCR)
print ('feature names: ')
print (datadigits.keys)
print ('target names :' )
print (datadigits.target_names)
print (datadigits.target)
print (datadigits.setdefault)



    





    





