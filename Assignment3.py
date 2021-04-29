import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("week3.csv",comment='#') 
X1=df.iloc[:, 0]
X2=df.iloc[:, 1]
X=np.column_stack((X1, X2))
y=df.iloc[:, 2]
y = np.array(df.iloc[:, 2])
y = np.reshape(y, (-1,1))

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title("Visualisation of Data", fontsize=14, pad=14)
ax.set_xlabel("X1 (Normalised)", fontsize=12, labelpad=12)
ax.set_ylabel("X2 (Normalised)", fontsize=12, labelpad=12)
ax.set_zlabel("y (Normalised)", fontsize=12, labelpad=12)
ax.scatter(X[:, 0], X[:, 1], y, c=y, edgecolor='black', linewidth=0.5, cmap='coolwarm')
plt.savefig('data_visualisation')
plt.show()

q = 5
Xpoly = PolynomialFeatures(q).fit_transform(X)

lassos = []
penalties = [1, 10, 1000]
for itr, penalty in enumerate(penalties):
    model = Lasso(alpha=1/(2*penalty))
    model.fit(Xpoly, y)
    lassos.append(model)
print('Parameter Values Table:')
print('%-8s %-21s %-20s' % ('Penalty', 'Intercept', 'Slope'))
for itr, model in enumerate(lassos):
    print('%-8s %-21a %-20a\n' % (penalties[itr], model.intercept_, model.coef_))
    
X_list =[]
X1_min, X1_max = X[:, 0].min() - 4, X[:, 0].max() + 4
X2_min, X2_max = X[:, 1].min() - 4, X[:, 1].max() + 4
X1_list = np.linspace(X1_min, X1_max, 30)
X2_list = np.linspace(X2_min, X2_max, 30)
for i in X1_list:
    for j in X2_list: 
        X_list.append([i ,j])
X_list = np.array(X_list)
Xpoly_list = PolynomialFeatures(q).fit_transform(X_list)

fig = plt.figure(figsize=(14, 14));
fig.tight_layout(pad=4.0)
for itr, model in enumerate(lassos):
    y_pred = model.predict(Xpoly_list)
    Xpoly_grid = np.reshape(Xpoly_list, (30,30,-1))
    y_pred = np.reshape(y_pred, (30,30))
    ax = fig.add_subplot(2, 2, itr + 1, projection='3d')
    ax.set_title("C = " + str(penalties[itr]), fontsize=14, pad=18)
    ax.set_xlabel("X1 (Normalised)", fontsize=12, labelpad=12)
    ax.set_ylabel("X2 (Normalised)", fontsize=12, labelpad=12)
    ax.set_zlabel("y (Normalised)", fontsize=12, labelpad=12)
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='coolwarm', edgecolor='black', linewidth=0.5)
    ax.plot_surface(Xpoly_grid[:,:,1], Xpoly_grid[:,:,2], y_pred ,cmap='coolwarm', edgecolor='none', alpha=0.75)
    legend_elements = [Line2D([0], [0], marker='o', markeredgecolor='black', linewidth=0.5, 
                              label='True', linestyle = 'None', markersize=10, alpha=0.75),
                       Line2D([0],[0], linewidth=2.0, label='Pred')]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.91,0.94))
plt.savefig('lasso_classifiers')
plt.show()

ridges = []
for itr, penalty in enumerate(penalties):
    model = Ridge(alpha=1/(2*penalty))
    model.fit(Xpoly, y)
    ridges.append(model)
print('Parameter Values Table:')
print('%-8s %-21s %-20s' % ('Penalty', 'Intercept', 'Slope'))
for itr, model in enumerate(ridges):
    print('%-8s %-21a %-20a\n' % (penalties[itr], model.intercept_, model.coef_))
    
fig = plt.figure(figsize=(14, 14));
fig.tight_layout(pad=4.0)
for itr, model in enumerate(ridges):
    y_pred = model.predict(Xpoly_list)
    Xpoly_grid = np.reshape(Xpoly_list, (30,30,-1))    
    y_pred = np.reshape(y_pred, (30,30))    
    ax = fig.add_subplot(2, 2, itr + 1, projection='3d')
    ax.set_title("C = " + str(penalties[itr]), fontsize=14, pad=18)
    ax.set_xlabel("X1 (Normalised)", fontsize=12, labelpad=12)
    ax.set_ylabel("X2 (Normalised)", fontsize=12, labelpad=12)
    ax.set_zlabel("y (Normalised)", fontsize=12, labelpad=12)
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='coolwarm', edgecolor='black', linewidth=0.5)
    ax.plot_surface(Xpoly_grid[:,:,1], Xpoly_grid[:,:,2], y_pred ,cmap='coolwarm', edgecolor='none', alpha=0.75)
    legend_elements = [Line2D([0], [0], marker='o', markeredgecolor='black', linewidth=0.5, 
                              label='True', linestyle = 'None', markersize=10, alpha=0.75),
                       Line2D([0],[0], linewidth=2.0, label='Pred')]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.91,0.94))
plt.savefig('ridge_classifiers')
plt.show()

means = []
variances = []
n_splits = [2, 5, 10, 25, 50, 100]
for n_split in n_splits:
    mses = []
    kf = KFold(n_splits=n_split)
    for train, test in kf.split(X):
        model = Lasso(alpha=1/(2*1))
        Xpoly = PolynomialFeatures(q).fit_transform(X[train])
        model.fit(Xpoly, y[train])
        Xpoly = PolynomialFeatures(q).fit_transform(X[test])
        ypred = model.predict(Xpoly)
        mse = (mean_squared_error(y[test], ypred))
        mses.append(mse)
    mean = np.mean(mses)
    means.append(mean)
    variance = np.var(mses)
    variances.append(variance)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Cross Validation of KFolds', fontsize=14)
ax.set_xlabel('Number of Folds', fontsize=12)
ax.set_ylabel('Mean (Normalised)', fontsize=12)
ax.errorbar(n_splits, means, yerr=np.sqrt(variances), linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(n_splits, means, marker='o', linewidth=1.0)
plt.savefig('cross_val_of_lasso_kfolds')
plt.show()

means = []
variances = []
penalties = [0.1, 0.5, 1, 5, 10, 50, 100]
for penalty in penalties:
    mses = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = Lasso(alpha=1/(2*penalty))
        Xpoly = PolynomialFeatures(q).fit_transform(X[train])
        model.fit(Xpoly, y[train])
        Xpoly = PolynomialFeatures(q).fit_transform(X[test])
        ypred = model.predict(Xpoly)
        mse = (mean_squared_error(y[test], ypred))
        mses.append(mse)
    mean = np.mean(mses)
    means.append(mean)
    variance = np.var(mses)
    variances.append(variance)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Cross Validation of C', fontsize=14)
ax.set_xlabel('C', fontsize=12)
ax.set_ylabel('Mean (Normalised)', fontsize=12)
ax.errorbar(penalties, means, yerr=np.sqrt(variances), linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(penalties, means, marker='o')
plt.savefig('cross_val_of_lasso_penalty')
plt.show()

means = []
variances = []
for penalty in penalties:
    mses = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = Ridge(alpha=1/(2*penalty))
        Xpoly = PolynomialFeatures(q).fit_transform(X[train])
        model.fit(Xpoly, y[train])
        Xpoly = PolynomialFeatures(q).fit_transform(X[test])
        ypred = model.predict(Xpoly)
        mse = (mean_squared_error(y[test], ypred))
        mses.append(mse)
    mean = np.mean(mses)
    means.append(mean)
    variance = np.var(mses)
    variances.append(variance)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Cross Validation of C', fontsize=14)
ax.set_xlabel('C', fontsize=12)
ax.set_ylabel('Mean (Normalised)', fontsize=12)
ax.errorbar(penalties, means, yerr=np.sqrt(variances), linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(penalties, means, marker='o')
plt.savefig('cross_val_of_ridge_penalty')
plt.show()