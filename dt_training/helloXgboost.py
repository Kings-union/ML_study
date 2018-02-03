
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]
seed = 7
test_size = 0.33
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)

# create the model by xgboost
model = XGBClassifier()

print(model)
eval_set = [(X_test,Y_test)]
model.fit(X_train,Y_train,early_stopping_rounds=10,eval_metric="logloss",eval_set=eval_set,verbose=True)
# plot_importance(model)
# pyplot.show()
Y_pred = model.predict(X_test)
predictions = [round(value) for value in Y_pred]
accuracy = accuracy_score(Y_test,predictions)
print("Accuracy: %.2f%%" % (accuracy*100.0))
