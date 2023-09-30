import time
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.extmath import safe_sparse_dot

class Bases:
    "building a library of candidate bases functions"
    @staticmethod
    def is_ok(val):
        """ Check if the input contains nan or inf.
        :param val: array
        :return: boolean (True/False)
        """
        if np.isnan(val).any() or np.isinf(val).any():
            return False
        else:
            return True

    def __init__(self,var,exponent,operators):
        'Initialise base class'
        '''
        All inputs in str
        :param var: list of variables 
        :param exponent: list of exponents
        :param operators: list of operators
        '''
        self.var = var
        self.exponent = exponent
        self.operators = operators

    def generate_bases(self,vars_dict):
        locals().update(vars_dict)
        vars = self.var
        exponents = self.exponent
        operators = self.operators

        'Generating list of univariate bases'
        B1 = []
        for var_i in vars:
            for exponent_i in exponents:
                b_exp = var_i + exponent_i
                if self.is_ok(eval(b_exp)):
                    B1.append(b_exp)
                    for operator_i in operators:
                        b_op = operator_i + '(' + b_exp + ')'
                        if self.is_ok(eval(b_op)):
                            B1.append(b_op)
        self.B1 = B1

      #Interact univariate bases to get non-univairate bases
        B2 = []
        B1 = self.B1
        for i in range(len(B1)):
            b_i = B1[i]
            for j in range(i - 1):
                b_j = B1[j]
                'Bool checks for not allowing op()*op()'
                bool = False
                for operator_i in self.operators:
                    if operator_i in b_j:
                        bool = True
                    else:
                        pass
                if bool == False:
                    b_inter = b_i + '*' + b_j
                    if self.is_ok(eval(b_inter)):
                        B2.append(b_inter)
        B = ['const'] + B1 + B2
        self.B2 = B2
        self.B = B

class ModelInference:
    def build_model(coefs, candidate_library):
        """ Writes mathematical model as string.

        :param coefs: Coefficient vector (n_features,)
        :param candidate_library: Symbolic library of candidate functions (n_features,)
        :return: model_string
        """
        i_nonzero = np.nonzero(coefs)[0]
        model_string = ''
        for i in i_nonzero:
            model_string = model_string + ' + ' + str(coefs[i]) + '*' + candidate_library[i]
        return model_string

    def discover_models(self,x_train,x_test,y_train,y_test,candidate_library):
        num_candidates = self.model_structures_.shape[-1]
        models = pd.DataFrame()
        self.ridge_alpha = 0
        for model_structure in self.model_structures_:
            index_nonzero = np.where(model_structure)[0]
            tmp_ = np.zeros(num_candidates)
            if self.ridge_alpha == 0:
                model = LinearRegression(fit_intercept=False,normalize=False)
            else:
                model = Ridge(alpha=self.ridge_alpha,fit_intercept=False,normalize=False)

            model.fit(x_train[:, index_nonzero], y_train)
            tmp_[index_nonzero] = model.coef_

            models = models.append({
                'coef_': tmp_,
                'mse_': np.average((y_test - safe_sparse_dot(x_test, tmp_)) ** 2.0),
                'complexity_': np.count_nonzero(tmp_),
                'l1_norm_': np.linalg.norm(tmp_, ord=1),
                'model': model,
                'model_structure_': model_structure,
                'ridge_alpha': self.ridge_alpha
            }, ignore_index=True)
        return models

class ModelDiscovery(Bases,ModelInference):
    """
    Model discovery class
    init to set defualt parameters and regularisation weight/ mixing parameter
    """
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
                if not all(eln.coef_ == 0.0):
                    model_structures.append(abs(eln.coef_) > 0.0)

        self.model_structures_ = np.unique(model_structures, axis=0)

