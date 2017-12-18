import pandas as pd
import numpy as np
import func_pandas

titanic_survival = pd.read_csv("titanic_train.csv")
titanic_survival.head()

# age = titanic_survival["Age"]
# # print(age.loc[0:10])
#
# age_is_null = pd.isnull(age)
# age_null_true = age[age_is_null]
# # print(age_null_true)
# age_null_count = len(age_null_true)
# # print(age_null_count)
#
# mean_age = sum(titanic_survival["Age"]) / len(titanic_survival["Age"])
# print(mean_age)
#
# good_ages = titanic_survival["Age"][age_is_null == False]
# correct_mean_age = sum(good_ages) / len(good_ages)
# print(correct_mean_age)
#
# correct_mean_age_pandas = titanic_survival["Age"].mean()
# print(correct_mean_age_pandas)

# passenger_class = [1, 2, 3]
# fares_by_class = {}
# print(fares_by_class)
# for this_class in passenger_class:
#     pclass_rows = titanic_survival[titanic_survival["Pclass"] == this_class]
#     pclass_fares = pclass_rows["Fare"]
#     fare_by_class = pclass_fares.mean()
#     fares_by_class[this_class] = fare_by_class
# print(fares_by_class)
#
# passenger_survival_pandas = titanic_survival.pivot_table(index="Pclass", values="Survived", aggfunc=np.mean)
# print(passenger_survival_pandas)
#
# # default func is np.mean
# passenger_age_pandas = titanic_survival.pivot_table(index="Pclass", values="Age")
# print(passenger_age_pandas)
#
# port_stats_pandas = titanic_survival.pivot_table(index="Embarked", values=["Fare", "Survived"], aggfunc=np.sum)
# print(port_stats_pandas)

# # drop the nan value row
# drop_na_columns = titanic_survival.dropna(axis=1)
# new_titanic_survival = titanic_survival.dropna(axis=0, subset=["Age", "Sex"])
# print(new_titanic_survival)

# # relocate the specific parameter
# row_index_83_age = titanic_survival.loc[83, "Age"]
# print(row_index_83_age)

new_titanic_survival = titanic_survival.sort_values("Age", ascending=False)
# print(new_titanic_survival[0:10])
titanic_reindexed = new_titanic_survival.reset_index(drop=True)
# print('_________________')
# print(titanic_reindexed.loc[0:10])


hundredth_row = titanic_reindexed.apply(func_pandas.hundredth_row)
print(hundredth_row)

# column_null_count = titanic_survival.apply(func_pandas.not_null_count)
# print(column_null_count)



