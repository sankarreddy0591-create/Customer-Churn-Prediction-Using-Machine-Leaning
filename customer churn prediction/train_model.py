
# Customer Churn Prediction Using Machine Learning

# import necessary libraries
import pandas as pd

# load the dataset
data = pd.read_csv('Data/customer_churn_data.csv')
print(" loaded dataset completed:", data.head())

# check for missing values
print("Missing values in each column:", data.isnull().sum())

# handle missing values if any
data["InternetService"]=data["InternetService"].fillna(data["InternetService"].mode()[0], inplace=True)

print("Missing values after handling:", data.isnull().sum())


# select features and target variable
x = data[["Age", "Gender", "Tenure", "MonthlyCharges"]]
y = data["Churn"]

print("Features and target variable selected.")
print("Features (x):", x.head())
print("Target variable (y):", y.head())

# encode categorical variables
x["Gender"] = x["Gender"].map({"Male": 0, "Female": 1})
# x["Gender"] = x["Gender"].apply(lambda x: 0 if x == "Male" else 1)
print("Categorical variables encoded.", x.head())

# encode target variable
y = y.map({"No": 0, "Yes": 1})
# y=y.apply(lambda x: 0 if x == "No" else 1)
print("Target variable encoded.", y.head()) 

# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Dataset split into training and testing sets.")
print("Training set size:", x_train.shape)
print("Testing set size:", x_test.shape)    
print("Training set target variable (y_train):", y_train.shape)
print("Training set target variable (y_test):", y_test.shape)

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("Feature scaling completed.")

# train the model using Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()    
model.fit(x_train_scaled, y_train)
print("Model training completed.")

# make predictions on the test set
y_pred = model.predict(x_test_scaled)
print("Predictions ",y_pred)

# model evaluation
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

print("\nClassification Report:", classification_report(y_test, y_pred))
print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred))

# save the trained model
import pickle
import os
os.makedirs('models', exist_ok=True)

with open('models/customer_churn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Model saved successfully.")