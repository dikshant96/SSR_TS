import tecplot as tp
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt

# datafile = '/home/dikshant/Desktop/Thesis - Datasets/CubeLES_processed.plt'
# dataset = tp.data.load_tecplot(datafile,read_data_option=2)
#
# SL1 = tp.data.extract.extract_slice(origin=(0.2,1,0),normal=(1,0,0))
# x = SL1.values(0)
# y = SL1.values(1)
# z = SL1.values(2)
# c = SL1.values(11)
#
# df_SL1_data = {'x':list(x), 'y':list(y), 'z':list(z), 'c':list(c)}
# df_SL1 = pd.DataFrame(data=df_SL1_data)

df_SL1 = pd.read_csv('SL1.csv')
df_SL1 = df_SL1.drop("Unnamed: 0",axis=1)

yi, zi = np.linspace(df_SL1['y'].min(),3.,500), np.linspace(-1.5,1.5,600)
yi, zi = np.meshgrid(yi,zi)

ci = interpolate.griddata((df_SL1['y'],df_SL1['z']),df_SL1['c'],(yi,zi),method='linear')
mask = np.empty(shape=np.shape(ci),dtype=bool)

for i in range(np.shape(yi)[0]):
    for j in range(np.shape(yi)[1]):
        mask[i,j] = False
        if (-0.5<=zi[i,j]<=0.5) & (0.>yi[i,j]):
            mask[i,j] = True

ci = np.ma.array(ci, mask=mask)


plt.imshow(ci,vmin=0.,vmax=df_SL1['c'].max(),origin='lower',extent=[yi.min(), 3., -1.5, 1.5])
plt.colorbar()
plt.show()


