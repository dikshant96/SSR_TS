import numpy as np
import pandas as pd
import sympy as sy

def is_ok(val):
    if np.isnan(val).any() or np.isinf(val).any():
        return False
    else:
        return True


def build_bases(vars,exps,funs):
    """Building univariate bases"""
    B1 = []
    for var in vars:
        for exp in exps:
            b_exp = var + exp
            if is_ok(eval(b_exp)):
                B1.append(b_exp)
    B11 = []

    """Applying functions to bases above"""
    if len(funs) != 0:
        for i in range(len(B1)):
            for fun in funs:
                b_func = fun + '(' + B1[i] + ')'
                if is_ok(eval(b_func)):
                    B11.append(b_func)
    B1 = B1 + B11
    return B1

def build_interactions(B1):
    """Generate interacting variable bases"""
    B2 = []
    for i in range(len(B1)):
        bi = B1[i]
        for j in range(i-1):
            bj = B1[j]
            b_inter = '('+bi+')*('+bj+')'
            if is_ok(eval(b_inter)):
                B2.append(b_inter)

    """Allows for up-to 3 non-linear interactions"""
    B3 = []
    for i in range(len(B2)):
        bi = B2[i]
        for j in range(len(B1)):
            bj = B1[j]
            b_inter = '('+bi+')*('+bj+')'
            if is_ok(eval(b_inter)):
                B3.append(b_inter)

    Btmp = ['const'] + B1 + B2 + B3
    return Btmp

def clean_bases(Btmp):
    # symplify bases
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

    # check for dublicates
    Bsym = list(set(Bsym))
    loopcount = len(Bsym)
    i = 0
    while i < loopcount:
        if isinstance(eval(Bsym[i]), int):
            del (Bsym[i])  # deletes all constant fields
            loopcount -= 1
        i += 1

    return np.array(sorted(Bsym))

z = np.linspace(-10, 0, 101)
const = np.ones(101)

# hidden process to be discovered
hidden_model_str = 'z**2.0 + z**2.0*np.sin(10.*z) + 0.123*z'
f_target = eval(hidden_model_str)

vars = ['const','z']
exps = ['**1','**2','**-1']
funs = ['abs','np.sin','np.cos']

B1 = build_bases(vars,exps,funs)
Btmp = build_interactions(B1)
Bsym = clean_bases(Btmp)