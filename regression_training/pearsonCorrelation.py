import func_regression

testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]

print("r^2:", func_regression.computeCorrelation(testX, testY)**2)

print(func_regression.polyfit(testX, testY, 1))