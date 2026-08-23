# Lab 3 — Simple Linear Regression
## Study Hours → Exam Score

### Dataset

The lab uses the Student Clean Dataset downloaded from the supplied Google
Drive file ID:

`1t5mmVocO1_fGXGqRftpekRom9yxq-ML4`

The loader in `slr_ols_student.py` performs the download and reads:

`student_clean_dataset.csv`

### SLR variables

- X = `Study_Hours`
- y = `Exam_Score`

### File responsibilities

#### `slr_ols_student.py`
Contains all function logic:

- `find_mean()`
- `find_slope_intercept()`
- `predict_lr()`
- `mae()`
- `mse()`
- `r_square()`
- `load_slr_dataset()`

#### Notebook
Contains:

- Instructions
- Explanations
- Student Tasks
- Driving code
- Visualization
- Residual analysis
- MAE / MSE / RMSE / R²
- Scikit-Learn LinearRegression
- Manual vs package comparison
- Final questions

No artificial data generator is used.
