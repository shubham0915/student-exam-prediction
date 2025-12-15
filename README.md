# 🎓 Student Exam Score Prediction

A machine learning project to predict student exam scores based on their habits and lifestyle factors.

## 📊 Features

- **5 ML Models**: Linear Regression, Polynomial Regression, Decision Tree, Random Forest, Gradient Boosting
- **Interactive Web App**: Built with Streamlit
- **Data Visualization**: Correlation heatmaps, feature importance charts, model comparison

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/student-exam-prediction.git
cd student-exam-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                          # Streamlit web application
├── student_prediction_clean.ipynb  # Jupyter notebook with analysis
├── student_habits_performance.csv  # Dataset
├── best_model.pkl                  # Trained model
├── label_encoders.pkl              # Categorical encoders
├── features_list.pkl               # Feature names
├── model_info.pkl                  # Model metadata
├── model_results.csv               # Model comparison results
├── feature_importance.csv          # Feature importance data
└── requirements.txt                # Python dependencies
```

## 📈 Model Performance

| Model | R² Score |
|-------|----------|
| Polynomial Regression | 89.60% |
| Gradient Boosting | 89.15% |
| Linear Regression | 88.88% |
| Random Forest | 87.79% |
| Decision Tree | 70.38% |

## 🔑 Key Insights

- **Study Hours** is the most important feature (70% importance)
- **Mental Health Rating** has significant impact on scores
- Polynomial Regression performs best with R² = 89.60%

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Plotly, Matplotlib, Seaborn

## 📝 License

This project is for educational purposes.
