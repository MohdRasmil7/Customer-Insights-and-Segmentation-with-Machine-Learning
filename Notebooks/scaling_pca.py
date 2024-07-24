# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:51:07 2024

@author: Muhammed Rasmil
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# Assuming X_train is your training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("pca.pkl", "wb") as pca_file:
    pickle.dump(pca, pca_file)
