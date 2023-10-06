import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.datasets import load_breast_cancer


from collections import Counter

data = load_breast_cancer(as_frame=True)

y_data = data.target
X_data = data.data
X_data.info()

corr_mat = X_data.corr(method = 'pearson')
high_cor = corr_mat>0.7
num_coll_pair = int((high_cor.sum().sum()-len(X_data.columns))/2)
print(f'Number of collinear pairs = {num_coll_pair}')

# mask = np.array(corr_mat)
# mask[np.tril_indices_from(mask)] = False
# plt.figure(figsize=(22,22), dpi = 100)
# sns.heatmap(corr_mat, mask = mask, annot=True, square=True, cmap='coolwarm')
# plt.show()
vif = pd.DataFrame()

vif["VIF_Factor"] = [variance_inflation_factor(X_data.values, i) for i in range(X_data.shape[1])]
vif["Features"] = X_data.columns