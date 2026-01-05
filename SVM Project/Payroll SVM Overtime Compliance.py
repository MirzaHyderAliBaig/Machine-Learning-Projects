
# Payroll Overtime Compliance SVM

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Employees Dataset

overtimePayroll = [
    {"employeeId": "0001","employeeName": "Amit Patel","hoursWeekly": 47,"provinceOfEmployment": "ON","overtimeApplied": 0,"compliance": 1},
    {"employeeId": "0002","employeeName": "Sarah Jones","hoursWeekly": 45,"provinceOfEmployment": "NS","overtimeApplied": 1,"compliance": 0},
    {"employeeId": "0003","employeeName": "Michael Chen","hoursWeekly": 50,"provinceOfEmployment": "BC","overtimeApplied": 0,"compliance": 1},
    {"employeeId": "0004","employeeName": "Priya Sharma","hoursWeekly": 40,"provinceOfEmployment": "AB","overtimeApplied": 0,"compliance": 0},
    {"employeeId": "0005","employeeName": "David Brown","hoursWeekly": 47,"provinceOfEmployment": "ON","overtimeApplied": 0,"compliance": 1},
    {"employeeId": "0007","employeeName": "Emily Davis","hoursWeekly": 52,"provinceOfEmployment": "NS","overtimeApplied": 0,"compliance": 1},
    {"employeeId": "0008","employeeName": "Rohit Verma","hoursWeekly": 50,"provinceOfEmployment": "AB","overtimeApplied": 0,"compliance": 1},
    {"employeeId": "0009","employeeName": "Jessica Lee","hoursWeekly": 40,"provinceOfEmployment": "BC","overtimeApplied": 0,"compliance": 0}
]

# using Dataframe for overtime payroll data.
df = pd.DataFrame(overtimePayroll)


# One-hot encode Province

encoder = OneHotEncoder(sparse_output=False)
province_encoded = encoder.fit_transform(df[['provinceOfEmployment']])
province_df = pd.DataFrame(province_encoded, columns=encoder.get_feature_names_out())

# Combine features for SVM
X = pd.concat([df[['hoursWeekly','overtimeApplied']], province_df], axis=1)
y = df['compliance']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Train Linear SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# ------------------------------
# Evaluation
# ------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 2D Decision Boundary (Projection)

# Using only hoursWeekly and overtimeApplied for visualization
X_plot = df[['hoursWeekly', 'overtimeApplied']].values
y_plot = df['compliance'].values

svm_2d = SVC(kernel='linear')
svm_2d.fit(X_plot, y_plot)

# Decision boundary line
w = svm_2d.coef_[0]
b = svm_2d.intercept_[0]
xx = np.linspace(X_plot[:,0].min()-1, X_plot[:,0].max()+1, 100)
yy = - (w[0]/w[1]) * xx - b/w[1]

# Margin lines
margin = 1/np.sqrt(np.sum(w**2))
yy_down = yy - (w[0]/w[1])*0 - margin/w[1]
yy_up = yy + (w[0]/w[1])*0 + margin/w[1]

# Plot
plt.figure(figsize=(8,6))
plt.scatter(X_plot[y_plot==0][:,0], X_plot[y_plot==0][:,1], color='blue', label='Compliant=0')
plt.scatter(X_plot[y_plot==1][:,0], X_plot[y_plot==1][:,1], color='red', label='Non-compliant=1')
plt.plot(xx, yy, 'k-', label='Decision Boundary')
plt.plot(xx, yy_down, 'k--', label='Margin')
plt.plot(xx, yy_up, 'k--')
plt.xlabel('Weekly Hours')
plt.ylabel('Overtime Applied')
plt.title('Linear SVM Decision Boundary for Overtime Compliance')
plt.legend()
plt.show()
