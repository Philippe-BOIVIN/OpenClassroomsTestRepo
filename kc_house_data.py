# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:55:19 2019

@author: boivi
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = 15, 10

df = pd.read_csv("kc_house_data.csv")

X = df[['bathrooms', 'bedrooms', 'condition', 'floors', 'grade', 'sqft_above', 'sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15', 'view', 'waterfront', 'yr_built', 'yr_renovated']]
Y = df['price']

X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.95, random_state=31)


#%% Régression linéaire

print("\n========================================")
print("Régression linéaire")
print("========================================\n")

# On crée un modèle de régression linéaire
lr = linear_model.LinearRegression(fit_intercept=True, normalize=True)

# On entraîne ce modèle sur les données d'entrainement
lr.fit(X_train,y_train)

lr_train_score = lr.score(X_train,y_train)
lr_test_score = lr.score(X_test,y_test)
print ("Training score : {0:.3f}".format(lr_train_score)) 
print ("Test score : {0:.3f}".format(lr_test_score))

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

n_alphas = 200
alphas = np.logspace(-2, 3, n_alphas)

ridge = Ridge(fit_intercept=True, normalize=True)

coefs = []
errors = []
for a in alphas:
    # On initialise le coefficient de pénalité
    ridge.set_params(fit_intercept=True, normalize=True, alpha=a)
    
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

print("\nMSE ridge : {0:.3f}".format(min(errors)[1]))

# Exemple Ridge pour alpha = 0.1
print("\nRidge alpha = 0.1")
ridge01 = Ridge(fit_intercept=True, normalize=True, alpha=0.1)
ridge01.fit(X_train,y_train)
train_score01=ridge01.score(X_train,y_train)
test_score01=ridge01.score(X_test,y_test)
print ("Training score pour alpha=0.1 : {0:.3f}".format(train_score01)) 
print ("Test score pour alpha =0.1: {0:.3f}".format(test_score01))

coef = pd.Series(ridge01.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour ridge : alpha = 0.01')
plt.show()

# Exemple Ridge pour alpha = 100
print("\nRidge alpha = 100")
ridge100 = Ridge(fit_intercept=True, normalize=True, alpha=100)
ridge100.fit(X_train,y_train)
train_score100=ridge100.score(X_train,y_train)
test_score100=ridge100.score(X_test,y_test)
print ("Training score pour alpha=100 : {0:.3f}".format(train_score100)) 
print ("Test score pour alpha =100 : {0:.3f}".format(test_score100))

coef = pd.Series(ridge100.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour ridge : alpha = 100')
plt.show()

# Exemple Ridge pour alpha = 1000
print("\nRidge alpha = 1000")
ridge1000 = Ridge(fit_intercept=True, normalize=True, alpha=1000)
ridge1000.fit(X_train,y_train)
train_score1000=ridge1000.score(X_train,y_train)
test_score1000=ridge1000.score(X_test,y_test)
print ("Training score pour alpha=1000 : {0:.3f}".format(train_score1000)) 
print ("Test score pour alpha =1000: {0:.3f}".format(test_score1000))

coef = pd.Series(ridge1000.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour ridge : alpha = 1000')
plt.show()


#%% Régression Lasso

print("\n========================================")
print("Régression lasso")
print("========================================\n")

n_alphas = 300
alphas = np.logspace(1, 4, n_alphas)
lasso = Lasso(fit_intercept=False)

coefs = []
errors = []
for a in alphas:
    lasso.set_params(fit_intercept=True, normalize=True, alpha=a, max_iter=10000)
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

print("\nMSE lasso : {0:.3f}".format(min(errors)[1]))


# Exemple lasso pour alpha = 100
print("\nLasso alpha = 100")
lasso100 = Lasso(fit_intercept=True, normalize=True, alpha=100, max_iter=10000)
lasso100.fit(X_train,y_train)
train_score100=lasso100.score(X_train,y_train)
test_score100=lasso100.score(X_test,y_test)
coeff_used100 = np.sum(lasso100.coef_!=0)
print ("Training score pour alpha=100: {0:.3f}".format(train_score100))
print ("Test score pour alpha =100: {0:.3f}".format(test_score100))
print ("Nombre de variables utilisées pour alpha=100: {0:.3f}".format(coeff_used100))

coef = pd.Series(lasso100.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour lasso : alpha = 100')
plt.show()

# Exemple lasso pour alpha = 1000
print("\nLasso alpha = 1000")
lasso1000 = Lasso(fit_intercept=True, normalize=True, alpha=1000, max_iter=10000)
lasso1000.fit(X_train,y_train)
train_score1000=lasso1000.score(X_train,y_train)
test_scor1000=lasso1000.score(X_test,y_test)
coeff_used1000 = np.sum(lasso1000.coef_!=0)
print ("Training score pour alpha=1000: {0:.3f}".format(train_score1000))
print ("Test score pour alpha =1000: {0:.3f}".format(test_score1000))
print ("Nombre de variables utilisées pour alpha=1000: {0:.3f}".format(coeff_used1000))

coef = pd.Series(lasso1000.coef_,X_train.columns).sort_values()
coef.plot(kind='bar', title='Coefficients pour lasso : alpha = 1000')
plt.show()
