# 🎓 Student Exam Score Prediction

Predict student exam scores from study habits and lifestyle factors using a clean ML pipeline plus a Streamlit app for live scoring.


## 📌 Overview
- Tabular dataset of student habits (study hours, sleep, mental health, etc.) mapped to exam scores
- Five regression models compared; **Polynomial Regression (degree 2) is the default best model**
- Reproducible preprocessing (label encoding, scaling) and train/test split with `random_state=42`
- Exported artifacts to reuse the best model outside the notebook

## ✨ Features
- Model zoo: Linear, Polynomial, Decision Tree, Random Forest, Gradient Boosting with side-by-side metrics
- Interactive Streamlit UI for real-time predictions and model selection
- Visuals: correlation heatmap, feature importance, model comparison bar chart
- Saved artifacts: `best_model.pkl`, `poly_features.pkl`, `label_encoders.pkl`, `features_list.pkl`, `model_info.pkl`, `model_results.csv`, `feature_importance.csv`

## 🚀 Quick Start
1) Clone
```bash
git clone https://github.com/YOUR_USERNAME/student-exam-prediction.git
cd student-exam-prediction
```
2) Install deps (use a virtualenv if you like)
```bash
pip install -r requirements.txt
```
3) Launch the app
```bash
streamlit run app.py
```

## 🧭 Using the App
1) Enter student habit inputs in the form
2) Keep the default **Best Model (Polynomial Regression)** or pick any model from the list
3) Click Predict to view the estimated exam score
4) Scroll to see model descriptions and metrics; refresh or change inputs to iterate quickly

## 🧪 Reproduce Training
- Open `student_prediction_clean.ipynb` and run all cells (random_state=42 for reproducibility)
- Outputs written alongside the notebook: `best_model.pkl`, `poly_features.pkl`, `label_encoders.pkl`, `features_list.pkl`, `model_info.pkl`, `model_results.csv`, `feature_importance.csv`
- Notebook compares five models and records metrics in `model_results.csv` for the leaderboard

## 📂 Project Structure
```
├── app.py                          # Streamlit web application
├── student_prediction_clean.ipynb  # Full analysis and model training
├── student_habits_performance.csv  # Dataset
├── best_model.pkl                  # Best trained model (Polynomial Regression)
├── poly_features.pkl               # PolynomialFeatures transformer
├── label_encoders.pkl              # Categorical encoders
├── features_list.pkl               # Feature names
├── model_info.pkl                  # Model metadata
├── model_results.csv               # Model comparison results
├── feature_importance.csv          # Feature importance data
├── requirements.txt                # Python dependencies
└── docs/
    └── images/                   # Place your screenshots here (see links above)
```

## 📈 Model Performance (test set, R²)
| Model | R² Score |
|-------|----------|
| Polynomial Regression | 89.60% |
| Gradient Boosting | 89.15% |
| Linear Regression | 88.88% |
| Random Forest | 87.79% |
| Decision Tree | 70.38% |

## 📊 Data Columns
- Numerical: Study Hours, Sleep Hours, Physical Activity, Mental Health Rating, Tutoring Sessions, Past Grades
- Categorical: Gender, Part-time Job, Internet Access, Extracurricular Activities, Parent Education, Exam Anxiety Level

## 🔑 Key Insights
- Study Hours dominates importance (≈70%); boosting focus on high-yield study time matters most
- Mental Health Rating has clear positive correlation with scores
- Polynomial Regression (degree 2) edges other models with highest R² and smooth generalization

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (regressors, preprocessing, metrics)
- Streamlit UI
- Plotly, Matplotlib, Seaborn for charts

## 📝 License
This project is for educational purposes.
