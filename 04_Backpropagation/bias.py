import pandas as pd

TrainData = pd.read_csv("./data/mnist_train.csv")
TestData = pd.read_csv("./data/mnist_test.csv")

X_train = TrainData.iloc[:,:-1]
X_test = TestData.iloc[:,:-1]

y_train = TrainData.iloc[:,-1]
y_test = TestData.iloc[:,-1]

print(X_train)