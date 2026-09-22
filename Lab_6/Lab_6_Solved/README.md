# Lab 5 — Final Student Package

## Core experiment
SLR using `Assignments` to predict `Exam_Score`.

The core comparison is:
**SLR / OLS vs SLR / Batch Gradient Descent**

Students compare slope, intercept, MAE, MSE, RMSE, and R² and inspect convergence.

## Extra / Miscellaneous
MLR with Batch Gradient Descent using:
- `Study_Hours`
- `Attendance`
- `Assignments`
- `Age`

Target: `Exam_Score`.

The MLR section is an independent extension. It is **not compared with SLR**.

## Clean input design
Pandas data is converted to NumPy once in the notebook. Numerical functions assume NumPy-array inputs and do not repeatedly call `np.asarray()`.
