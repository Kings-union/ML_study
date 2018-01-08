import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

train_df = pd.read_csv('./input/train.csv', index_col=0)
test_df = pd.read_csv('./input/test.csv', index_col=0)

train_df.head()

# print(train_df.head())

# # oragnise the data
# prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
# prices.hist()

# pop the result from train sets from the integrate data
Y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pd.concat((train_df, test_df), axis=0)
all_df.shape

# print(all_df.shape

all_df['MSSubClass'].dtypes # result should be int64

# change the type to string
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
#
# print(all_df['MSSubClass'].value_counts())

# using pandas get_dummies for the One-Hot
all_dummy_df = pd.get_dummies(all_df)
# print(all_dummy_df.head())

# check the null value
# print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))

mean_cols = all_dummy_df.mean()
# print(mean_cols.head(10))
all_dummy_df = all_dummy_df.fillna(mean_cols)
print(all_dummy_df.isnull().sum().sum())

numeric_cols = all_df.columns[all_df.dtypes != 'object']
print(numeric_cols)

#standarize the data
numeric_cols_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_cols_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_cols_means) / numeric_cols_std

# Moduling
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

print (dummy_train_df.shape, dummy_test_df.shape)

X_train = dummy_train_df.values
X_test = dummy_test_df.values

# LR moduling
alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, Y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))


# plt.plot(alphas, test_scores)
# plt.title("Alpha vs CV Error")
# plt.show()

# Random Forest Moduling

max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, Y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error")
plt.show()

# Ensemble

ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
ridge.fit(X_train, Y_train)
rf.fit(X_train, Y_train)

y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))

y_final = (y_ridge + y_rf) / 2

submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})

submission_df.head(10)