import time
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.extmath import safe_sparse_dot
import sympy as sy

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

    def __init__(self,vars,exponent,operators):
        'Initialise base class'
        '''
        All inputs in str
        :param var: list of variables 
        :param exponent: list of exponents
        :param operators: list of operators
        '''
        self.vars = vars
        self.exponent = exponent
        self.operators = operators

    def get_active_vars(self,vars_dict,vars_active):
        vars_cull = []
        for i,var in enumerate(self.vars):
            var_active_i = vars_active[i]
            if var_active_i == 1.:
                vars_cull.append(var)
            else:
                pass
        self.vars = vars_cull

    def generate_bases(self,vars_dict):
        locals().update(vars_dict)
        vars = self.vars
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

        Btmp = ['const'] + B1 + B2

        Bsym = []
        for i in range(len(Btmp)):
            tmp = Btmp[i].replace('np.exp', 'exp')
            tmp = tmp.replace('np.log10', 'log10')
            tmp = tmp.replace('np.sin', 'sin')
            tmp = tmp.replace('np.cos', 'cos')
            tmp = str(sy.simplify(sy.sympify(tmp)))
            tmp = tmp.replace('exp', 'np.exp')
            tmp = tmp.replace('log10', 'np.log10')
            tmp = tmp.replace('sin', 'np.sin')
            tmp = tmp.replace('cos', 'np.cos')
            Bsym.append(tmp)


        Bsym = list(set(Bsym))

        # loopcount = len(Bsym)
        # i = 0
        # while i < loopcount:
        #     if isinstance(eval(Bsym[i]), int):
        #         del (Bsym[i])  # deletes all constant fields
        #         loopcount -= 1
        #     i += 1
        #print(Bsym)
        B = sorted(Bsym)

        self.B2 = B2
        self.B = B

    def simplify_clib_vectors(self,c_models):
        csym = []
        for i in range(len(c_models)):
            tmp = c_models[i].replace('np.exp', 'exp')
            tmp = tmp.replace('np.log10', 'log10')
            tmp = tmp.replace('np.sin', 'sin')
            tmp = tmp.replace('np.cos', 'cos')
            tmp = str(sy.simplify(sy.sympify(tmp)))
            tmp = tmp.replace('exp', 'np.exp')
            tmp = tmp.replace('log10', 'np.log10')
            tmp = tmp.replace('sin', 'np.sin')
            tmp = tmp.replace('cos', 'np.cos')
            csym.append(tmp)

        csym = list(set(csym))
        csym = sorted(csym)
        return csym

    def get_complexity(self,models):
        complexity = np.zeros((len(models),))
        for i,b in enumerate(models):
            b = b.replace('**','^')
            all_comps = b.split('*')
            n_comps = len(all_comps)
            ops_str = []
            for op in self.operators:
                if op in b:
                    ops_str.append(op)
            n_ops = len(ops_str)
            n_comps = n_comps + n_ops
            complexity[i] = n_comps
        return complexity
class ModelInference:
    @staticmethod
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
        model_string = model_string.replace('np.exp', 'exp')
        model_string = model_string.replace('np.log10', 'log10')
        model_string = model_string.replace('np.sin', 'sin')
        model_string = model_string.replace('np.cos', 'cos')
        #model_string = str(sy.sympify(sy.sympify(model_string)))

        return model_string

    def discover_models(self,ridge_alphas,x_train,x_test,y_train,y_test, candidate_library):
        num_candidates = self.model_structures_.shape[-1]
        models = pd.DataFrame()
        for alpha in ridge_alphas:
            for model_structure in self.model_structures_:
                index_nonzero = np.where(model_structure)[0]
                tmp_ = np.zeros(num_candidates)
                if alpha == 0:
                    model = LinearRegression(fit_intercept=False,normalize=False)
                else:
                    model = Ridge(alpha=alpha,fit_intercept=False,normalize=False)

                model.fit(x_train[:, index_nonzero], y_train)
                tmp_[index_nonzero] = model.coef_

                models = models.append({
                    'coef_': tmp_,
                    'str_': self.build_model(np.round(tmp_,5),candidate_library),
                    'mse_': np.average((y_test - safe_sparse_dot(x_test, tmp_)) ** 2.0),
                    'complexity_': np.count_nonzero(tmp_),
                    'l1_norm_': np.linalg.norm(tmp_, ord=1),
                    'model': model,
                    'model_structure_': model_structure,
                    'ridge_alpha': alpha
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
            print(f'Ran ({time.time()-start:3.2g}s) alpha={alpha:3.2g}')
        self.model_structures_ = np.unique(model_structures, axis=0)

