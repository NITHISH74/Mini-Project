# Ex-10-Mini-Project:
### Date:
## Aim:
To perform feature encoding and scaling on a dataset, specifically using the Iris dataset, to prepare the data for Data analysis tasks.
## Explanation:
The provided Python program preprocesses the Iris dataset for a classification task in six steps. It loads the dataset, separates features and target variables, performs label encoding on the target variable, and standardizes the features using standard scaling. The encoded target variable (`y_train_encoded` and `y_test_encoded`) and scaled features (`X_train_scaled` and `X_test_scaled`) are then ready for use in a machine learning model.

## Algorithms:
### Step 1:
Import Libraries.
### Step 2: 
Load the iris Dataset.
### Step 3: 
Display the Original Iris Dataset.
### Step 4: 
Split Data into Features and Target.
### Step 5:
Feature Encoding and Scaling.
### Step 6: 
Display Encoded Target and Scaled Features.




## Code:
```
Developed by : NITHISHWAR S
Register No. : 212221230071
```
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the Iris dataset (or replace with your own dataset)
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

data
data.shape
data.info()

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

X.shape

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature encoding: Label encoding for the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Display the encoded target variable
print("\nEncoded Target Variable:")
y_test_encoded

# Feature scaling: Standardization using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the scaled features
print("\nScaled Features (X_train):")
print(X_train_scaled[:5])
```
## Output:
### Dataset:
![image](https://github.com/NITHISH74/Mini-Project/assets/94164665/36546f9c-5289-4990-a1b5-6d919d357da9)
### Data information:
![image](https://github.com/NITHISH74/Mini-Project/assets/94164665/8619f6f6-1fbb-4e6a-9bc8-f310a3363205)
### Encoded Target Variable:
![image](https://github.com/NITHISH74/Mini-Project/assets/94164665/dfa5f371-680b-49c1-84cb-18ef489f9d37)
### Scaled Features:
![image](https://github.com/NITHISH74/Mini-Project/assets/94164665/792e8ce2-b8c5-44c6-9029-b5849470713c)


## Result:
The program preprocesses the Iris dataset by encoding the target variable using Label Encoder, scaling the features using StandardScaler, and splitting the data into training and testing sets.
