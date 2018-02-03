from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

data_set = loadtxt('pima-indians-diabetes.csv',delimiter=",")
X = data_set[:,0:8]
Y = data_set[:,8]

model = XGBClassifier()
learning_rate = [0.0001,0.001,0.01,0.1,0.2,0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
grid_search = GridSearchCV(model,param_grid,scoring="neg_log_loss",n_jobs=-1,cv=kfold)
grid_result = grid_search.fit(X,Y)

print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean,param in zip(means,params):
    print("%f with: %r" % (mean,param))