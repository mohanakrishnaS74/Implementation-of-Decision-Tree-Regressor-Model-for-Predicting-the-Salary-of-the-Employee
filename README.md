# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and separate input feature (Level) and target variable (Salary).

2.Split the dataset into training and testing sets.

3.Train a Decision Tree Regressor model using the training data.

4.Predict salary values and evaluate model performance using error metrics. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: mohanakrishna S 
RegisterNumber:212225040253
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv(r"C:\Users\acer\Downloads\Salary.csv")

X = df[["Level"]].values
y = df["Salary"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Level"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Regressor for Employee Salary Prediction")
plt.show()


new_exp = [[5]]
predicted_salary = model.predict(new_exp)

print("Predicted Salary for 5 years experience:", predicted_salary[0])
```
## Output:
<img width="1246" height="100" alt="560148141-0029e89f-075c-464d-b4ed-df7c2c1d7017" src="https://github.com/user-attachments/assets/fb5959c0-db41-4018-a2ed-087dad4cc52e" />
<img width="1240" height="690" alt="560148219-c3436ff3-df74-45e0-8fa1-5e46ef4bb81f" src="https://github.com/user-attachments/assets/ba8619ce-57cc-4d34-bbe7-7e9b115bf363" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
