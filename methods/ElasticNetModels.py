import time

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model._coordinate_descent import _alpha_grid

class ElasticNetFunctions:

    def __init__(self,n_alphas,l1_ratios):
        self.l1_ratios = l1_ratios
        self.max_iter = 1000
        self.fit_intercept = False
        self.warm_start = True
        self.standardisation = True
        self.n_alphas = n_alphas


    def get_alpha_grid(self,x,y):
        self.alphas = _alpha_grid(x,y,n_alphas=self.n_alphas,l1_ratio=0.5)


    def get_EN_models(self,x,y):
        model_structures = []
        for alpha in self.alphas:
            start = time.time()
            for l1_ratio in self.l1_ratios:
                eln = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,fit_intercept=self.fit_intercept,warm_start=self.warm_start,max_iter=self.max_iter)
                eln.fit(x,y)
                print(eln.coef_)
                if not all(eln.coef_ == 0.0):
                    model_structures.append(abs(eln.coef_) > 0.0)

        self.model_structures_ = np.unique(model_structures, axis=0)

