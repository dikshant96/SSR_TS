import numpy as np
import pandas as pd
import networkx as nx

class Clique:
    def __init__(self,X_data,threshold=0.9):
        self.X_data = X_data
        self.threshold = threshold
        self.get_cliques()

    def get_cliques(self):
        R = np.corrcoef(self.X_data, rowvar = False)
        R = np.where(np.abs(R) > self.threshold, True, False)

        G = nx.from_numpy_matrix(R)
        cliques = []
        working = list(nx.find_cliques(G))
        while len(working) > 0:
            working.sort(key=len)
            largest_clique = working[-1]
            cliques.append(largest_clique)
            [G.remove_node(node) for node in largest_clique]
            working = list(nx.find_cliques(G))
        self.cliques = cliques

    def get_cliqued_models(self,models,models_complexity):
        ind_cliqued = []
        for clique in self.cliques:
            models_clique = models[clique[:]]
            complexity = models_complexity[clique[:]]
            ind_complexity = np.argsort(complexity)
            ind_models = clique[ind_complexity[0]]
            ind_cliqued.append(ind_models)
        models_cliqued = models[ind_cliqued]
        self.ind_cliqued = ind_cliqued
        return models_cliqued

    def clique_data(self,X_train,X_train_scaled,X_test):
        X_train_cliqued = X_train[:,self.ind_cliqued]
        X_test_cliqued = X_test[:,self.ind_cliqued]
        X_train_scaled_cliqued = X_train_scaled[:,self.ind_cliqued]
        return X_train_cliqued, X_train_scaled_cliqued, X_test_cliqued