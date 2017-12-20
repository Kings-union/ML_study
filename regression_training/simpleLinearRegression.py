import func_regression

x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]

b0, b1 = func_regression.fitSLR(x, y)

print("intercept:", b0, " slope:", b1)

x_test = 6

y_test = func_regression.predict(6, b0, b1)

print("y_test:", y_test)