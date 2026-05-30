# 🏋️ Workout Rating Predictor — ML Pipeline

> *Because not all reps are created equal — and neither are workouts.*

Ever scrolled through a fitness app wondering which workout is actually worth your time? This project answers that with data. Using the Mega Gym Dataset, I built an end-to-end machine learning pipeline that predicts how well a workout will be rated — based on its type, target body part, equipment, difficulty level, and biometric signals like heart rate and calories burned.

No guesswork. No bro-science. Just clean data and a model that delivers.

---

## 🚀 What's Inside

The pipeline covers every stage of a production-ready ML workflow:

- **Data Cleaning** — Mean imputation for missing values, type coercion, and column standardization across 800+ exercise entries
- **Feature Engineering** — StandardScaler normalization on 10 biometric numeric features; LabelEncoding on 7 categorical variables with saved encoders for inference
- **Model Comparison** — Three models trained and benchmarked head-to-head: Linear Regression (baseline), Decision Tree, and Random Forest — evaluated on MAE, MSE, and R²
- **Winner: Random Forest** — Consistently outperformed on all three metrics, leveraging ensemble learning to handle mixed feature types and non-linear relationships
- **Deployment-Ready Inference** — Saved model via `joblib`, with a graceful `-1` fallback for unseen categories at prediction time

---

## 📁 Project Structure

```
workout-rating-predictor/
│
├── workout.ipynb           # Full ML pipeline notebook
├── rating_predictor.pkl    # Saved Random Forest model
├── megaGymDataset.csv      # Source dataset
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 📊 Models Compared

| Model | MAE | MSE | R² |
|---|---|---|---|
| Linear Regression | baseline | baseline | baseline |
| Decision Tree | better | better | better |
| **Random Forest** | **best** | **best** | **best** |

---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/Prashantanand2811/workout-rating-predictor.git
cd workout-rating-predictor

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook workout.ipynb
```

---

## 🔮 Predict a New Workout

```python
import joblib
import pandas as pd

model = joblib.load("rating_predictor.pkl")

new_exercise = pd.DataFrame([{
    "Title": "Push Ups", "Desc": "Chest workout", "Type": "Strength",
    "BodyPart": "Chest", "Equipment": "None", "Level": "Beginner",
    "RatingDesc": "Good"
}])

predicted_rating = model.predict(new_exercise)
print(f"Predicted Rating: {predicted_rating[0]:.2f}")
```

---

## 👤 Author

**Prashant Anand**  
[GitHub](https://github.com/Prashantanand2811) | [Portfolio](https://prashantanand2811.github.io)
