# Overtime Compliance SVM

This project uses a **Linear Support Vector Machine** to predict employee overtime compliance using simulated Canadian payroll data.

## Dataset
- `data/canadian_overtime_dataset.csv` contains:
  - EmployeeID, EmployeeName, Province, WeeklyHours, OvertimeApplied, Compliance

## Features
- `hoursWeekly` (numeric)
- `overtimeApplied` (0/1)
- `Province` (one-hot encoded)

## Model
- Linear SVM (no kernel)
- Predicts compliance (0 = Compliant, 1 = Non-compliant)
- Decision boundary visualization included in `figures/decision_boundary.png`

## Instructions
1. Install required packages:
```bash
pip install pandas scikit-learn matplotlib
