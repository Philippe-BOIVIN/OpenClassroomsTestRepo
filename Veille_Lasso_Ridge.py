# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:13:42 2019

@author: boivi
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

plt.rcParams['figure.figsize'] = 5, 5

raw_data = pd.read_csv('prostate_dataset.txt', delimiter='\t')

X_train = raw_data.iloc[:60,1:-3]
y_train = raw_data.iloc[:60,-2]
X_test = raw_data.iloc[60:,1:-3]
y_test = raw_data.iloc[60:,-2]

#%% Régression linéaire

print("\n========================================")
print("Régression linéaire")
print("========================================\n")

# On crée un modèle de régression linéaire
lr = linear_model.LinearRegression()

# On entraîne ce modèle sur les données d'entrainement
lr.fit(X_train,y_train)

lr_train_score = lr.score(X_train,y_train)
lr_test_score = lr.score(X_test,y_test)
print ("Training score : {0:.3f}".format(lr_train_score)) 

# On récupère l'erreur de norme 2 sur le jeu de données test comme baseline
lr_error = np.mean((lr.predict(X_test) - y_test) ** 2)

print("\nMSE : {0:.3f}".format(lr_error))

coef = pd.Series(lr.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour la régression linéaire')
plt.show()

#%% Régression Ridge

print("\n========================================")
print("Régression ridge")
print("========================================\n")

# Exemple Ridge pour alpha = 0.1
print("\nRidge alpha = 0.1")
ridge01 = Ridge(alpha=0.1)
ridge01.fit(X_train,y_train)
train_score01=ridge01.score(X_train,y_train)
test_score01=ridge01.score(X_test,y_test)
print ("Training score pour alpha=0.1 : {0:.3f}".format(train_score01)) 

coef = pd.Series(ridge01.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour ridge : alpha = 0.01')
plt.show()

# Exemple Ridge pour alpha = 10
print("\nRidge alpha = 10")
ridge10 = Ridge(alpha=10)
ridge10.fit(X_train,y_train)
train_score10=ridge10.score(X_train,y_train)
test_score10=ridge10.score(X_test,y_test)
print ("Training score pour alpha=10 : {0:.3f}".format(train_score10)) 

coef = pd.Series(ridge10.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour ridge : alpha = 10')
plt.show()

# Exemple Ridge pour alpha = 100
print("\nRidge alpha = 100")
ridge100 = Ridge(alpha=100)
ridge100.fit(X_train,y_train)
train_score100=ridge100.score(X_train,y_train)
test_score100=ridge100.score(X_test,y_test)
print ("Training score pour alpha=100 : {0:.3f}".format(train_score100)) 

coef = pd.Series(ridge100.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour ridge : alpha = 100')
plt.show()


#%% Ridge : variation alpha
 
# Variation du coefficient de pénalité alpha
n_alphas = 200
alphas = np.logspace(-4, 5, n_alphas)

ridge = Ridge()

coefs = []
errors = []
for a in alphas:
    # On initialise le coefficient de pénalité
    ridge.set_params(alpha=a)
    
    # On entraine
    ridge.fit(X_train, y_train)
    
    # On stocke les coefficients et la MSE de la prédiction pour l'alpha courant
    coefs.append(ridge.coef_)
    errors.append([lr_error, np.mean((ridge.predict(X_test) - y_test) ** 2)])

# Plot des coeffs en fonction de alpha
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('valeur des coefficients')
plt.title('Ridge : coefficients en fonction du coefficient de pénalité de la régularisation')
plt.show()

# Plot de l'erreur en fonction de alpha
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.show()

print("\nmin MSE ridge : {0:.3f}".format(min(errors)[1]))



#%% Régression Lasso

print("\n========================================")
print("Régression lasso")
print("========================================\n")


# Exemple lasso pour alpha = 0.01
print("\nLasso alpha = 0.01")
lasso001 = Lasso(alpha=0.01)
lasso001.fit(X_train,y_train)
train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)
print ("Training score pour alpha=0.01: {0:.3f}".format(train_score001))
print ("Nombre de variables utilisées pour alpha=0.01: {0:.3f}".format(coeff_used001))

coef = pd.Series(lasso001.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour lasso : alpha = 0.01')
plt.show()

# Exemple lasso pour alpha = 0.1
print("\nLasso alpha = 0.1")
lasso01 = Lasso(alpha=0.1)
lasso01.fit(X_train,y_train)
train_score01=lasso01.score(X_train,y_train)
test_score01=lasso01.score(X_test,y_test)
coeff_used01 = np.sum(lasso01.coef_!=0)
print ("Training score pour alpha=0.1: {0:.3f}".format(train_score01))
print ("Nombre de variables utilisées pour alpha=0.1: {0:.3f}".format(coeff_used01))

coef = pd.Series(lasso01.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour lasso : alpha = 0.1')
plt.show()

# Exemple lasso pour alpha = 1
print("\nLasso alpha = 1")
lasso1 = Lasso(alpha=1)
lasso1.fit(X_train,y_train)
train_score1=lasso1.score(X_train,y_train)
test_score1=lasso1.score(X_test,y_test)
coeff_used1 = np.sum(lasso1.coef_!=0)
print ("Training score pour alpha=1: {0:.3f}".format(train_score1))
print ("Nombre de variables utilisées pour alpha=1: {0:.3f}".format(coeff_used1))

coef = pd.Series(lasso1.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour lasso : alpha = 1')
plt.show()


#%% Lasso : variation alpha

# Variation du coefficient de pénalité alpha
n_alphas = 300
alphas = np.logspace(-4, 1, n_alphas)

lasso = Lasso(fit_intercept=False)

coefs = []
errors = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    y_pred = lasso.predict(X_test)
    errors.append([lr_error, np.mean((y_pred - y_test) ** 2)])

# Plot des coeffs en fonction de alpha
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('valeur des coefficients')
plt.title('Lasso : coefficients en fonction du coefficient de pénalité de la régularisation')
plt.show()

# Plot de l'erreur en fonction de alpha
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.show()

print("\nmin MSE lasso : {0:.3f}".format(min(errors)[1]))



#statisticsformachinelearning.pdf
#Data Science Fondamentaux et études de cas-Eyrolles.epub
#


