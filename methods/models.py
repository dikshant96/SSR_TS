import numpy as np
import pandas as pd

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
