# Predicting-Workout-Ratings-with-Machine-Learning
Workout Rating Predictor — ML Pipeline

Because not all reps are created equal — and neither are workouts.

Ever scrolled through a fitness app wondering which workout is actually worth your time? This project answers that with data. Using the Mega Gym Dataset, I built an end-to-end machine learning pipeline that predicts how well a workout will be rated — based on its type, target body part, equipment, difficulty level, and biometric signals like heart rate and calories burned.
No guesswork. No bro-science. Just clean data and a model that delivers.

What's Inside
The pipeline covers every stage of a production-ready ML workflow:

Data Cleaning — Mean imputation for missing values, type coercion, and column standardization across 800+ exercise entries
Feature Engineering — StandardScaler normalization on 10 biometric numeric features; LabelEncoding on 7 categorical variables with saved encoders for inference
Model Comparison — Three models trained and benchmarked head-to-head: Linear Regression (baseline), Decision Tree, and Random Forest — evaluated on MAE, MSE, and R²
Winner: Random Forest — Consistently outperformed on all three metrics, leveraging ensemble learning to handle mixed feature types and non-linear relationships
Deployment-Ready Inference — Saved model via joblib, with a graceful -1 fallback for unseen categories at prediction time


Files
FileDescriptionworkout.ipynbFull ML pipeline notebookrating_predictor.pklSaved Random Forest modelmegaGymDataset.csvSource datasetREADME.mdThis file
