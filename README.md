# 📊 Customer-Churn-Prediction-Using-Machine-Leaning
Customer Churn



## 📊 Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that analyzes customer information and predicts whether a customer is likely to **churn (leave the service)** or **stay with the company**.

The project includes:

* Data loading
* Data understanding
* Exploratory Data Analysis (EDA)
* Data cleaning
* Data visualization
* Feature selection
* Data preprocessing
* Model training
* Model evaluation
* Model comparison

Several Machine Learning classification algorithms are tested to identify a suitable model for customer churn prediction.

---

## 🎯 Project Objective

The main objective of this project is to:

* Understand customer behavior.
* Analyze factors related to customer churn.
* Identify patterns between churn and customer characteristics.
* Build Machine Learning classification models.
* Compare different classification algorithms.
* Predict whether a customer will churn or not.

---

## 🧠 Problem Statement

Customer retention is important for businesses because losing existing customers can affect revenue and growth.

This project uses historical customer data to identify patterns associated with churn.

The target variable is:

* **Churn = Yes** → Customer has churned.
* **Churn = No** → Customer has not churned.

The Machine Learning models learn from existing customer data and classify customers into these two categories.

---

# 🔄 Project Workflow

```text
📁 Customer Churn Dataset
          ↓
📥 Load Dataset
          ↓
🔍 Understand Dataset
          ↓
🧹 Data Cleaning
          ↓
📊 Exploratory Data Analysis
          ↓
📈 Data Visualization
          ↓
🎯 Feature Selection
          ↓
🔢 Convert Categorical Data
          ↓
✂️ Train-Test Split
          ↓
⚖️ Feature Scaling
          ↓
🤖 Train Machine Learning Models
          ↓
📏 Evaluate Models
          ↓
🔄 Compare Models
          ↓
🏆 Select Suitable Model
```

---

# 1️⃣ Import Required Libraries

The project starts by importing the required Python libraries.

### Libraries Used

* **NumPy** – Numerical operations
* **Pandas** – Data loading and data manipulation
* **Matplotlib** – Data visualization
* **Scikit-learn** – Machine Learning algorithms and evaluation metrics

Warnings are also suppressed to keep the notebook output clean.

---

# 2️⃣ Load the Dataset

The customer churn CSV dataset is loaded into a Pandas DataFrame.

The dataset contains customer-related information such as:

* Age
* Gender
* Tenure
* Monthly Charges
* Contract Type
* Internet Service
* Churn
* And other customer attributes

The dataset is then stored in a Pandas DataFrame for further analysis.

---

# 3️⃣ Understand the Dataset

Several DataFrame functions are used to understand the structure and characteristics of the dataset.

### Dataset inspection includes:

* First five records
* Last five records
* Dataset shape
* Dataset information
* Statistical summary
* Column names
* Missing-value analysis
* Duplicate-value analysis

### Functions Used

```text
head()
tail()
shape
info()
describe()
columns
isna()
duplicated()
```

This step helps understand the dataset before performing Machine Learning.

---

# 4️⃣ Data Cleaning

Missing values are checked using the `isna()` function.

The `InternetService` column contains missing values.

These missing values are handled by replacing them with an empty string.

After cleaning, the dataset is checked again to confirm the missing-value status.

Duplicate records are also checked to identify repeated observations.

---

# 5️⃣ Correlation Analysis

The numerical columns are selected from the dataset.

A correlation matrix is generated to understand relationships between numerical variables.

Correlation values generally range from:

```text
-1  → Strong negative relationship
 0  → No linear relationship
+1  → Strong positive relationship
```

Correlation analysis helps understand how numerical variables are related to each other.

---

# 6️⃣ Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand customer behavior and identify patterns related to churn.

Several visualizations and group-based analyses are performed.

---

## 📌 Churn Distribution

##<img width="390" height="410" alt="Churn (Yes or No)" src="https://github.com/user-attachments/assets/d4eeadc6-3458-4a6b-8642-634ca8e441dc" />

The number of customers in each churn category is analyzed.

A pie chart is created to visualize:

* Customers who churned
* Customers who did not churn

This helps understand the overall distribution of the target variable.

---

## 📌 Average Monthly Charges by Churn

##<img width="563" height="463" alt="Average Monthly Charges By Churn" src="https://github.com/user-attachments/assets/4b982146-bf13-47f2-ac11-6b33b8c2bfab" />


The average `MonthlyCharges` is calculated separately for customers who churned and customers who did not churn.

A bar chart is used to visualize the comparison.

### Business Insight

Customers with different monthly charge levels may show different churn behavior.

This can help the business understand whether pricing or monthly charges are associated with customer churn.

---

## 📌 Average Monthly Charges by Churn and Gender

##<img width="642" height="454" alt="Average Monthly Charges By Churn and Gender" src="https://github.com/user-attachments/assets/07641874-0abb-48d9-ac98-b5ce0f5497ec" />


The average monthly charges are analyzed based on:

* Churn
* Gender

A horizontal bar chart is used to compare the groups.

This provides a more detailed view of monthly charges across different customer segments.

---

## 📌 Average Tenure by Churn

##<img width="390" height="389" alt="Average Tenure by Churn" src="https://github.com/user-attachments/assets/4d91db5f-fd92-46da-9761-b8d107b23885" />


The average customer tenure is calculated for each churn category.

Tenure represents how long a customer has been associated with the company.

This analysis helps identify whether customer loyalty or customer duration is related to churn.

---

## 📌 Average Age by Churn



The average age of customers is analyzed according to their churn status.

This helps identify potential differences in customer age between churned and retained customers.

---

## 📌 Average Monthly Prices by Contract Type

##<img width="563" height="552" alt="Contract Type Average Price" src="https://github.com/user-attachments/assets/b8a8e1fb-1c03-44f5-93b2-932e18c619f1" />


The average monthly charges are calculated for different contract types.

A bar chart is created to compare the average prices across contract categories.

This helps understand the relationship between contract type and customer charges.

---

## 📌 Monthly Charges Distribution

##<img width="552" height="434" alt="Histogram of Monthly Charges" src="https://github.com/user-attachments/assets/c1b1e253-f0ae-4f07-8ac6-e0e711100014" />


A histogram is created for `MonthlyCharges`.

A histogram shows the distribution of numerical values across different ranges.

### Histogram Components

* **Counts** → Number of observations in each bin
* **Bins** → Value ranges
* **Patches** → Histogram bars

---

## 📌 Tenure Distribution

##<img width="552" height="434" alt="Histogram Of Tenure" src="https://github.com/user-attachments/assets/c6deea64-3952-4b9a-a66e-9a7e55361233" />


A histogram is also created for the `Tenure` variable.

This helps understand how customer tenure is distributed across the dataset.

---

# 7️⃣ Feature Selection

For the Machine Learning models, selected customer attributes are used as input features.

### Input Features

```text
Age
Gender
Tenure
MonthlyCharges
```

### Target Variable

```text
Churn
```

The problem is therefore treated as a **Binary Classification Problem**.

---

# 8️⃣ Categorical Data Encoding

Machine Learning algorithms require numerical input.

Therefore, categorical values are converted into numerical values.

### Gender

```text
Female → 1
Male   → 0
```

### Churn

```text
Yes → 1
No  → 0
```

This converts the categorical variables into numerical format suitable for Machine Learning models.

---

# 9️⃣ Train-Test Split

The dataset is divided into two parts:

* **Training Data** → Used to train the Machine Learning models.
* **Testing Data** → Used to evaluate model performance.

An **80:20 split** is used.

```text
80% → Training Data
20% → Testing Data
```

A random state is used to make the experiment reproducible.

Stratified splitting is also used based on the churn target to maintain the class distribution between training and testing data.

---

# 🔟 Feature Scaling

Feature scaling is performed using **StandardScaler**.

Scaling transforms numerical features to a comparable scale.

The scaler is:

* Fitted on training data
* Applied to training data
* Applied to testing data

This approach helps prevent information from the test dataset from influencing the training process.

Scaling is used for models such as:

* Logistic Regression
* Support Vector Classifier
* K-Nearest Neighbors

---

# 1️⃣1️⃣ Machine Learning Models

Multiple classification algorithms are trained and evaluated.

The following models are implemented:

### 1. Logistic Regression

Logistic Regression is a classification algorithm used to predict binary outcomes.

In this project, it predicts:

```text
0 → No Churn
1 → Churn
```

---

### 2. Random Forest Classifier

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees.

It can capture nonlinear relationships between customer characteristics and churn.

---

### 3. Support Vector Classifier (SVC)

SVC attempts to find an optimal decision boundary between different classes.

It is used to classify customers into churn and non-churn categories.

---

### 4. K-Nearest Neighbors (KNN)

KNN predicts the class of a customer based on the classes of nearby observations.

In this project, the number of neighbors is set to:

```text
K = 5
```

---

### 5. Gradient Boosting Classifier

Gradient Boosting builds models sequentially, where each new model attempts to improve the errors made by previous models.

It is used to evaluate another powerful classification approach for churn prediction.

---

# 1️⃣2️⃣ Model Evaluation

The trained models are evaluated using classification metrics.

### Metrics Used

#### Accuracy

Accuracy represents the overall percentage of correctly classified observations.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

#### Precision

Precision measures how many predicted positive cases were actually positive.

For this project, it indicates how accurately the model identifies customers predicted to churn.

---

#### Recall

Recall measures how many actual churn customers were correctly identified by the model.

Recall is particularly important in churn prediction because missing a customer who is likely to leave can be costly to a business.

---

#### F1-Score

F1-score combines precision and recall into a single metric.

It is useful when evaluating classification performance, especially when class distributions are not perfectly balanced.

---

#### Confusion Matrix

A confusion matrix shows the number of:

* True Positives
* True Negatives
* False Positives
* False Negatives

It provides a detailed view of model predictions.

---

# 1️⃣3️⃣ Model Comparison

The performance of the Machine Learning algorithms is compared.

| Model               | Evaluation                                          |
| ------------------- | --------------------------------------------------- |
| Logistic Regression | Accuracy + Classification Report + Confusion Matrix |
| Random Forest       | Accuracy + Classification Report + Confusion Matrix |
| SVC                 | Accuracy + Classification Report                    |
| KNN                 | Accuracy + Classification Report                    |
| Gradient Boosting   | Accuracy + Classification Report                    |

The models are compared based on their classification performance rather than relying only on accuracy.

---

# 1️⃣4️⃣ Key Analysis Findings

The EDA provides several useful observations about customer behavior.

### Important areas analyzed:

* Churn distribution
* Monthly charges
* Customer tenure
* Customer age
* Gender
* Contract type
* Numerical feature relationships

The analysis shows how customer characteristics can be explored to identify patterns associated with churn.

The Machine Learning stage then uses selected customer features to classify churn behavior.

---

# 💼 Business Applications

A Customer Churn Prediction system can help businesses:

* Identify customers at risk of leaving.
* Improve customer retention strategies.
* Develop targeted offers.
* Understand customer behavior.
* Reduce customer acquisition costs.
* Improve customer satisfaction.
* Prioritize high-risk customers for retention campaigns.

---

# 🛠️ Technologies Used

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| Python           | Programming Language    |
| Pandas           | Data Manipulation       |
| NumPy            | Numerical Computing     |
| Matplotlib       | Data Visualization      |
| Scikit-learn     | Machine Learning        |
| Jupyter Notebook | Development Environment |

---

# 📂 Project Structure

```text
Customer Churn Prediction/
│
├── Data/
│   └── customer_churn_data.csv
│
├── Notebook/
│   └── Customer_Churn_Prediction.ipynb
│
├── model/
│   └── trained_model.pkl
│
├── output
│   └── images/
│         ├── Average Monthly Charges By Churn and Gender.png.png
│         ├── Average Monthly Charges by Churn.png
│         ├── Average Tenure by Churn.png
│         ├── Chun (Yes o No).png
│         ├── Contract Type Average Price.png
│         ├── Histogram of Monthly Charges.png
│         └── Histogam Of Tenue.png
│
│
├── README.md
│
└── requirements.txt
```

---

# 🚀 End-to-End Summary

```text
1. Load Customer Churn Dataset
        ↓
2. Inspect Dataset
        ↓
3. Check Missing Values
        ↓
4. Handle Missing Values
        ↓
5. Check Duplicate Records
        ↓
6. Perform Correlation Analysis
        ↓
7. Perform Exploratory Data Analysis
        ↓
8. Create Data Visualizations
        ↓
9. Select Relevant Features
        ↓
10. Encode Categorical Variables
        ↓
11. Split Data into Train and Test Sets
        ↓
12. Apply Feature Scaling
        ↓
13. Train Classification Models
        ↓
14. Generate Predictions
        ↓
15. Evaluate Models
        ↓
16. Compare Model Performance
        ↓
17. Identify a Suitable Model
```

---

# 📌 Conclusion

This project demonstrates a complete Machine Learning workflow for **Customer Churn Prediction**.

The project starts with raw customer data and progresses through data cleaning, exploratory analysis, visualization, feature engineering, model training, and model evaluation.

Multiple classification algorithms are tested and compared to understand their effectiveness in predicting customer churn.

The project provides practical experience in applying Machine Learning techniques to a real-world business problem where predicting customer churn can support better customer retention decisions.

---

## 👨‍💻 Project Type

**Data Science | Machine Learning | Classification | Customer Analytics**

### 🎯 Main Goal

**Predict customer churn and identify customer behavior patterns using Machine Learning.**
