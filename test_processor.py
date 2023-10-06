import numpy as np
from OF_Funcs.OFPy import OFPy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from methods.SparseSymbolicRegression import Bases,ModelDiscovery,ModelInference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from time import time
from statsmodels.tools.tools import add_constant
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

I,Iv = get_invariants(S,W,gradc_n)
Vb = get_vector_bases(S,W,gradc_n)

vars = ['i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11','i12','i13','i14']
exponents = ['**-1','**-0.5','**0.5','**1','**2']
# operators = ['np.sin','np.cos','abs']
operators = ['abs']
const = np.ones((6000,3,1))

vars_dict = {'i1':Iv[:,0],'i2':Iv[:,1],'i3':Iv[:,2],'i4':Iv[:,3],'i5':Iv[:,4],'i6':Iv[:,5],
             'i7':Iv[:,6],'i8':Iv[:,7],'i9':Iv[:,8],'i10':Iv[:,9],'i11':Iv[:,10],'i12':Iv[:,11],
             'i13':Iv[:,12],'i14':Iv[:,13],'const':const}

vectors = ['vb1','vb2','vb3','vb4','vb5','vb6']
vectors_dict = {'vb1':Vb[:,0],'vb2':Vb[:,1],'vb3':Vb[:,2],
                'vb4':Vb[:,3],'vb5':Vb[:,5],'vb6':Vb[:,5]}

locals().update(vars_dict)
locals().update(vectors_dict)

vars_active = np.ones(len(vars))

for i,var in enumerate(vars):
    inv_i = vars_dict[var]
    if inv_i.all() == 0:
        vars_active[i] = 0.

vars_active = np.array([0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0.])
vectors_active = [1,1,0,1,0,0]

vectors_n = []
for i, vector in enumerate(vectors):
    if vectors_active[i] == 1:
        vectors_n.append(vector)

vectors = vectors_n

model_bases = Bases(vars,exponents,operators)
model_bases.get_active_vars(vars_dict,vars_active)
model_bases.generate_bases(vars_dict)

C_models = build_and_eval_Clib(model_bases.B,vectors)
C_models = model_bases.simplify_clib_vectors(C_models)

#C_models = C_models[0:300]

D = np.stack([eval(C_models[i]) for i in range(len(C_models))])

D = np.reshape(D,(len(C_models),6000*3))
D = D.T
#
# start = time()
# vif = pd.DataFrame()
# vif["VIF_Factor"] = [variance_inflation_factor(D,i) for i in range(D.shape[1])]
# end = time()
# print(f'Time take for 100 models VIF {end-start}')
y = ciDelta
y2 = np.reshape(y,(6000*3,))

# X_train, X_test, y_train, y_test = train_test_split(D, y, random_state=1)
# scaler = StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)

# models_discovered = ModelDiscovery(10,[0.01,0.1,0.2,0.5,0.7,0.9,0.95,0.99,1.0])
# models_discovered.get_alpha_grid(X_train_scaled,y_train)
#models_discovered.get_EN_models(X_train_scaled,y_train)