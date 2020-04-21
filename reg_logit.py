# -*- coding: utf-8 -*-
import pandas as pd 
from sklearn.linear_model import LogisticRegression as logit 
import statsmodels.api as sm
#import matplotlib.pyplot as plt

df = pd.read_csv("emailOf.csv") 
#Selection de la variable catégorique dépendante 
df["TookAction"] = df["TookAction"].astype('category')
Y_dependent = df["TookAction"].cat.codes

#Création de dummies variable, toujours omettre une dummy variable(ici "Female")
# Si male et 1 => C'est un homme | Si male et 0 => C'est une femme 
gender_dummy = pd.get_dummies(df["Gender"])
X_independent = pd.concat([(df["Age"]),gender_dummy["Male"]],1)

#Modèle de Reg Logistique avec scikit-learn 
mylogit1 = logit(penalty= 'l2', solver='newton-cg')
mylogit1.fit(X_independent,Y_dependent)
#Affichage 
print("coef : ", mylogit1.coef_)

#Modèle de Reg Logistique avec scikit-learn 
mylogit2 = sm.Logit(Y_dependent,X_independent)
#Affichage 
print(mylogit2.fit().summary())
