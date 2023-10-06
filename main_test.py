import networkx as nx
import numpy as np
import pandas as pd
import sympy as sy
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import enet_path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from methods.SparseSymbolicRegression import Bases,ModelDiscovery,ModelInference
from methods.Clique import Clique
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA

z = np.linspace(-10, 0, 101)
const = np.ones(101)

vars = ['z']
exponents = ['**1','**2']
operators = ['np.sin','np.cos','abs']
vars_dict = {'z':z,'const':const}

model_bases = Bases(vars,exponents,operators)
model_bases.generate_bases(vars_dict)
model_bases.get_complexity()

# hidden process to be discovered
hidden_model_str = 'z**2.0 + z**2.0*np.sin(10.*z) + 0.123*z'
f_target = eval(hidden_model_str)

B_data = np.stack([eval(model_bases.B[i]) for i in range(len(model_bases.B))]).T


x_train, x_test, y_train, y_test = train_test_split(B_data,f_target,random_state=1)
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)


models = np.array(model_bases.B)
models_complexity = np.array(model_bases.complexity)

clique = Clique(x_train_scaled)

models_cliqued = clique.get_cliqued_models(models,models_complexity)
X_train, X_train_scaled, X_test = clique.clique_data(x_train,x_train_scaled,x_test)

models_discovered = ModelDiscovery(100,[0.01,0.1,0.2,0.5,0.7,0.9,0.95,0.99,1.0])
models_discovered.get_alpha_grid(X_train_scaled,y_train)
models_discovered.get_EN_models(X_train_scaled,y_train)
models = models_discovered.discover_models(X_train,X_test,y_train,y_test,models_cliqued)
