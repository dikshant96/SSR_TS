import numpy as np
from OF_Funcs.OFPy import OFPy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from methods.SparseSymbolicRegression import Bases,ModelDiscovery,ModelInference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
from methods.Clique import Clique
from math import nan
def is_ok(val):
    """ Check if the input contains nan or inf.
    :param val: array
    :return: boolean (True/False)
    """
    if np.isnan(val).any() or np.isinf(val).any():
        return False
    else:
        return True

def get_invariants(S,W,v):
    n = nx*ny*nz

    I = np.zeros((n,14))
    Iv = np.zeros((n,14,3))
    for i in range(n):
        Si = S[i,:]
        Wi = W[i,:]
        vi = v[i,:]
        Si2 = Si @ Si
        Wi2 = Wi @ Wi
        SiWi = Si @ Wi
        I[i,0] = np.trace(Si)
        I[i,1] = np.trace(Si2)
        I[i,2] = np.trace(Si @ Si2)
        I[i,3] = np.trace(Wi2)
        I[i,4] = np.trace(Si @ Wi2)
        I[i,5] = np.trace(Si2 @ Wi2)
        I[i,6] = np.trace(Si2 @ Wi2 @ SiWi)
        I[i,7] = vi @ vi
        I[i,8] = vi @ Si @ vi
        I[i,9] = vi @ Si2 @ vi
        I[i,10] = vi @ Wi2 @ vi
        I[i,11] = vi @ SiWi @ vi
        I[i,12] = vi @ Si2 @ Wi @ vi
        I[i,13] = vi @ Wi @ Si @ Wi2 @ vi
    Iv[:,:,0] = I
    Iv[:,:,1] = I
    Iv[:,:,2] = I
    Iv = Iv.reshape((n,14,3,1))
    return I, Iv

def get_vector_bases(S,W,v):
    n = nx*ny*nz
    Vb = np.zeros((n,6,3,1))

    for i in range(n):
        Si = S[i,:]
        Wi = W[i,:]
        vi = np.reshape(v[i,:],(3,1))
        Vb[i,0,:] = vi
        Vb[i,1,:] = Si @ vi
        Vb[i,2,:] = Si @ Si @ vi
        Vb[i,3,:] = Wi @ vi
        Vb[i,4,:] = Wi @ Wi @ vi
        Vb[i,5,:] = (Si @ Wi + Wi @ Si) @ vi

    return Vb



def build_and_eval_Clib(B,Vb):
    C = []
    for vbi in Vb:
        for bi in B:
            cbi = bi + '*' + vbi
            C.append(cbi)
    C = np.array(C)
    return C

'''Read OF files'''
"Reading OpenFOAM folder for frozen AKN"
home_dir = str(Path.home())
of_dir = home_dir + '/OpenFOAM/dikshant-7/run/'
case_dir = of_dir + 'channel180_AKN_Frozen/'

files = {
    'C':'C', #cell centres
    'U':'U', #velocity
    'V':'V', #cell volumes
    'T':'T', #cell scalar/ temp,
    'gradU':'grad(U)',
    'k':'k',
    'epsilon':'epsilon',
    'gradc':'grad(T)',
    'kt':'kt',
    'epsilont':'epsilont',
    'ciDelta':'ciDelta',
    'ciBoussinesq':'ciBoussinesq'
}

channel180 = OFPy(case_dir, files, mesh=0)

nx = 60
ny = 100
nz = 1

U = channel180.read_internalvector(channel180.fnames[1])
c = channel180.read_internalscalar(channel180.fnames[3])
gradU = channel180.read_internaltensor(channel180.fnames[4])
gradU_t = np.reshape(gradU, (len(gradU),3,3))

k = channel180.read_internalscalar(channel180.fnames[5])
epsilon = channel180.read_internalscalar(channel180.fnames[6])
gradc = channel180.read_internalvector(channel180.fnames[7])
kt = channel180.read_internalscalar(channel180.fnames[8])
epsilont = channel180.read_internalscalar(channel180.fnames[9])

ciDelta = channel180.read_internalvector(channel180.fnames[10])
ciB = channel180.read_internalvector(channel180.fnames[11])


'''Get Invariants'''
S = np.zeros((nx*ny*nz,3,3))
W = np.zeros((nx*ny*nz,3,3))
gradc_n = np.zeros((nx*ny*nz,3))

for i in range(nx*ny):
    gradU_ti = gradU_t[i].T
    S[i,:,:] = 0.5*(gradU_ti + gradU_ti.T)*(epsilon[i]/k[i])
    S[i,:,:] = S[i,:,:] - 1./3. * np.eye(3) * np.trace(S[i,:,:])
    W[i,:,:] = 0.5*(gradU_ti - gradU_ti.T)*(epsilon[i]/k[i])
    gradc_n[i,:] = gradc[i,:] * (k[i]**(3/2)/epsilon[i]) * kt[i]**-0.5

gradc_n[abs(gradc_n) < 1e-10] = 0.
S[abs(S)<1e-10] = 0.
W[abs(W)<1e-10] = 0.

y_wall = channel180.y

y_dist = []
for yi_wall in y_wall:
    if yi_wall > 1:
        yi_wall = 2-yi_wall
        y_dist.append(yi_wall)
    else:
        y_dist.append(yi_wall)
y_dist = np.array(y_dist).reshape(6000,1)

Red = y_dist[:,]*(k[:,]**0.5)
Red_mat = np.tile(Red, (3)).reshape(6000,3,1)

I,Iv = get_invariants(S,W,gradc_n)
Vb = get_vector_bases(S,W,gradc_n)

vars = ['i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11','i12','i13','i14','i15']
exponents = ['**-1','**-0.5','**0.5','**1','**2']
# operators = ['np.sin','np.cos','abs']
operators = ['abs','np.exp']
const = np.ones((6000,3,1))

vars_dict = {'i1':Iv[:,0],'i2':Iv[:,1],'i3':Iv[:,2],'i4':Iv[:,3],'i5':Iv[:,4],'i6':Iv[:,5],
             'i7':Iv[:,6],'i8':Iv[:,7],'i9':Iv[:,8],'i10':Iv[:,9],'i11':Iv[:,10],'i12':Iv[:,11],
             'i13':Iv[:,12],'i14':Iv[:,13],'i15':Red_mat,'const':const}

vectors = ['vb1','vb2','vb3','vb4','vb5','vb6']
vectors_dict = {'vb1':Vb[:,0],'vb2':Vb[:,1],'vb3':Vb[:,2],
                'vb4':Vb[:,3],'vb5':Vb[:,5],'vb6':Vb[:,5]}

locals().update(vars_dict)
locals().update(vectors_dict)

models_f = pd.read_csv('models_channel_full.csv')
query_inds = models_f.query("ridge_alpha==0.0").index

models = models_f.loc[query_inds]


mse = []
maxe = []
for i in range(len(models)):
    model_best = models.iloc[i]
    model_str = model_best['str_']
    if isinstance(model_str, float):
        mse.append(1000)
        maxe.append(1000)
    else:
        model_str = model_str.replace('exp', 'np.exp')
        ciDelta_inf = eval(model_str)
        ciDelta_inf = ciDelta_inf.reshape((6000, 3))
        ciDelta_inf = ciDelta_inf * ((k ** 0.5) * (kt ** 0.5))
        x_y = channel180.x[30::60]
        y_y = channel180.y[30::60]
        ciDelta_y = ciDelta[30::60]
        ciDelta_inf_y = ciDelta_inf[30::60]
        mse_i = np.average((ciDelta_y - ciDelta_inf_y)**2)
        maxe_i = np.average((ciDelta_y - ciDelta_inf_y) ** 2)
        mse.append(mse_i)
        maxe.append(maxe_i)

idx = np.argmin(mse)
model_best = models.iloc[idx]
model_str = model_best['str_']
model_str = model_str.replace('exp','np.exp')
ciDelta_inf = eval(model_str)
ciDelta_inf = ciDelta_inf.reshape((6000, 3))
ciDelta_inf = ciDelta_inf * ((k ** 0.5) * (kt ** 0.5))
x_y = channel180.x[20::60]
y_y = channel180.y[20::60]
ciDelta_y = ciDelta[20::60]
ciDelta_inf_y = ciDelta_inf[20::60]

plt.figure(1)
plt.plot(y_y,ciDelta_y[:,1],'-o')
plt.plot(y_y,ciDelta_inf_y[:,1],'-x')

plt.figure(2)
plt.plot(y_y,ciDelta_y[:,0],'-o')
plt.plot(y_y,ciDelta_inf_y[:,0],'-x')


# '''Clean up vars'''
# vars_active = np.ones(len(vars))
# for i,var in enumerate(vars):
#     inv_i = vars_dict[var]
#     if inv_i.all() == 0:
#         vars_active[i] = 0.
#
# #vars_active = np.array([0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0.,1.])
# vectors_active = [1,1,0,1,0,0]
#
# vectors_n = []
# for i, vector in enumerate(vectors):
#     if vectors_active[i] == 1:
#         vectors_n.append(vector)
#
# vectors = vectors_n
#
# model_bases = Bases(vars,exponents,operators)
# model_bases.get_active_vars(vars_dict,vars_active)
# model_bases.generate_bases(vars_dict)
#
# C_models = build_and_eval_Clib(model_bases.B,vectors)
# C_models = model_bases.simplify_clib_vectors(C_models)
#
# D = np.stack([eval(C_models[i]) for i in range(len(C_models))])
# D = np.reshape(D,(len(C_models),6000*3))
# D = D.T
#
# y = ciDelta/((k**0.5) * (kt**0.5))
# y = np.reshape(y,(6000*3,))
#
# x_train, x_test, y_train, y_test = train_test_split(D,y,random_state=1)
# scaler = StandardScaler().fit(x_train)
# x_train_scaled = scaler.transform(x_train)
#
# models = np.array(C_models)
# models_complexity = model_bases.get_complexity(C_models)
#
# cliques = Clique(x_train_scaled)
# models_cliqued = cliques.get_cliqued_models(models,models_complexity)
# X_train, X_train_scaled, X_test = cliques.clique_data(x_train,x_train_scaled,x_test)
#
# models_discovered = ModelDiscovery(100,[0.01,0.1,0.2,0.5,0.7,0.9,0.95,0.99,1.0])
# models_discovered.get_alpha_grid(X_train_scaled,y_train)
# models_discovered.get_EN_models(X_train_scaled,y_train)
# models_inf = models_discovered.discover_models([0,0.001,0.01,0.1],X_train,X_test,y_train,y_test,models_cliqued)