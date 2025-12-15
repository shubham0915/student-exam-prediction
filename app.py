import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import os
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for neon styling
st.markdown("""
<style>
    /* Dark background - OCEAN THEME */
    .stApp {
        background: linear-gradient(135deg, #1a3a4a 0%, #1d4157 30%, #1a4560 60%, #1d5070 100%);
    }
    
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #0099cc, #00ffcc, #00d4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
        margin-bottom: 10px;
    }
    
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #7ec8e3;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Neon box container */
    .neon-box {
        background: rgba(10, 30, 50, 0.8);
        border: 2px solid #00d4ff;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3),
                    inset 0 0 20px rgba(0, 212, 255, 0.1);
        margin: 20px 0;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #00d4ff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Selectbox label */
    .stSelectbox label {
        color: #00ffcc !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #0099cc, #00b8d4) !important;
        color: #0a1a2a !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        padding: 12px 30px !important;
        border: none !important;
        border-radius: 30px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5),
                    0 0 40px rgba(0, 153, 204, 0.3) !important;
        width: 100% !important;
        margin-top: 10px !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.8),
                    0 0 60px rgba(0, 153, 204, 0.5) !important;
    }
    
    /* Result display */
    .result-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 153, 204, 0.1));
        border: 2px solid #00d4ff;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        margin-top: 15px;
    }
    
    .score-label {
        color: #7ec8e3;
        font-size: 1rem;
        margin-bottom: 5px;
    }
    
    .score-value {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #00ffcc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Comparison box */
    .compare-box {
        background: linear-gradient(135deg, rgba(0, 255, 204, 0.1), rgba(0, 212, 255, 0.1));
        border: 2px solid #00ffcc;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 255, 204, 0.3);
        margin-top: 15px;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: rgba(10, 30, 50, 0.9);
        border: 1px solid #00d4ff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .rec-title {
        color: #00d4ff;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .rec-item {
        color: #e0e0e0;
        padding: 8px 0;
        border-bottom: 1px solid rgba(0, 255, 136, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(10, 30, 50, 0.8);
        border: 1px solid #00d4ff;
        border-radius: 10px;
        color: #00d4ff;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff, #0099cc) !important;
        color: #0a1a2a !important;
    }
    
    /* Metric card */
    .metric-card {
        background: rgba(10, 30, 50, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 20px 0 10px 0;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
    }
    
    hr {
        border: 1px solid rgba(0, 212, 255, 0.3);
        margin: 15px 0;
    }
    
    /* FIX: Make all text visible */
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #ffffff !important;
    }
    
    /* Metric styling - FIXED for visibility */
    [data-testid="stMetric"] {
        background: rgba(10, 30, 50, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.4);
        border-radius: 10px;
        padding: 15px;
    }
    
    [data-testid="stMetricLabel"] {
        color: #00d4ff !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p, span, label, .stText {
        color: #e0e0e0 !important;
    }
    
    /* Tab content text */
    .stTabs [data-baseweb="tab-panel"] {
        color: #ffffff !important;
    }
    
    /* Selectbox and slider text */
    .stSelectbox > div > div {
        color: #ffffff !important;
        background-color: rgba(10, 30, 50, 0.9) !important;
    }
    
    /* Selectbox dropdown options */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(10, 30, 50, 0.95) !important;
        border-color: #00d4ff !important;
    }
    
    /* Dropdown menu options */
    [data-baseweb="popover"] {
        background-color: #0a1a2a !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #0a1a2a !important;
    }
    
    [data-baseweb="menu"] li {
        color: #ffffff !important;
        background-color: #0a1a2a !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #0d3050 !important;
        color: #00d4ff !important;
    }
    
    /* Info, success, warning boxes */
    .stAlert > div {
        color: #ffffff !important;
    }
    
    /* DataFrame text */
    .stDataFrame {
        color: #ffffff !important;
    }
    
    /* Column text */
    [data-testid="column"] {
        color: #ffffff !important;
    }
    
    /* Make write text white */
    .element-container {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model and related files
@st.cache_resource
def load_model_and_data():
    model = joblib.load("best_model.pkl")
    
    # Try to load additional files if they exist
    model_info = None
    feature_importance = None
    le_dict = None
    
    if os.path.exists("model_info.pkl"):
        model_info = joblib.load("model_info.pkl")
    if os.path.exists("feature_importance.csv"):
        feature_importance = pd.read_csv("feature_importance.csv")
    if os.path.exists("label_encoders.pkl"):
        le_dict = joblib.load("label_encoders.pkl")
    
    # Load original data for insights
    df = pd.read_csv("student_habits_performance.csv")
    
    # Load model results for model selection
    model_results = None
    if os.path.exists("model_results.csv"):
        model_results = pd.read_csv("model_results.csv")
    
    return model, model_info, feature_importance, le_dict, df, model_results

# Function to create and train a specific model
@st.cache_resource
def get_trained_model(model_name, _df_model, _features):
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import PolynomialFeatures
    
    X = _df_model[_features]
    y = _df_model['exam_score']
    
    poly = None
    
    if "Linear Regression" in model_name:
        trained_model = LinearRegression()
        trained_model.fit(X, y)
    elif "Polynomial" in model_name:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        trained_model = LinearRegression()
        trained_model.fit(X_poly, y)
    elif "Decision Tree" in model_name:
        trained_model = DecisionTreeRegressor(max_depth=5, min_samples_split=2, random_state=42)
        trained_model.fit(X, y)
    elif "Random Forest" in model_name:
        trained_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        trained_model.fit(X, y)
    elif "Gradient Boosting" in model_name:
        trained_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        trained_model.fit(X, y)
    else:
        # Default to best model
        trained_model = joblib.load("best_model.pkl")
    
    return trained_model, None, poly

model, model_info, feature_importance, le_dict, df, model_results = load_model_and_data()

# Header
st.markdown('<h1 class="main-title">🎓 Student Exam Score Predictor</h1>', unsafe_allow_html=True)

# Display model info if available
if model_info:
    col1, col2, col3 = st.columns(3)
    with col1:
        # Clean model name - remove number prefix and (Advanced)/(Unit) suffixes
        raw_model_name = model_info.get("model_name", "ML Model")
        import re
        clean_model_name = re.sub(r'^\d+\.\s*', '', raw_model_name)  # Remove "7. " prefix
        clean_model_name = re.sub(r'\s*\(.*?\)\s*$', '', clean_model_name)  # Remove "(Advanced)" etc.
        st.metric("🤖 Model", clean_model_name)
    with col2:
        st.metric("📊 Accuracy (R²)", f"{model_info.get('r2_score', 0.85)*100:.1f}%")
    with col3:
        st.metric("📉 Error (RMSE)", f"±{model_info.get('rmse', 10):.1f}")

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predict Score", "⚖️ Compare Scenarios", "📊 Data Insights", "💡 Feature Analysis", "📈 Model Comparison"])

# ============== TAB 1: PREDICTION ==============
with tab1:
    # Model Selection Section
    st.markdown('<p class="section-header">🤖 Select Prediction Model</p>', unsafe_allow_html=True)

    # Available models list (3 Syllabus + 2 Advanced = 5 Models)
    available_models = [
        "🏆 Best Model (Default - Polynomial Regression)",
        "1. Linear Regression (Syllabus)",
        "2. Polynomial Regression (Syllabus)",
        "3. Decision Tree (Syllabus)",
        "4. Random Forest (Advanced)",
        "5. Gradient Boosting (Advanced)"
    ]
    
    selected_model = st.selectbox(
        "Choose a model for prediction:",
        available_models,
        index=0,
        help="Select a specific model or use the default best model"
    )
    
    # Show model info based on selection
    if model_results is not None and selected_model != "🏆 Best Model (Default - Polynomial Regression)":
        model_key = selected_model.split(". ", 1)[1] if ". " in selected_model else selected_model
        matching_rows = model_results[model_results['model'].str.contains(model_key.split(" (")[0], case=False, na=False)]
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("R² Score", f"{row['R2']*100:.1f}%")
            with mcol2:
                st.metric("RMSE", f"{row['RMSE']:.2f}")
            with mcol3:
                st.metric("MAE", f"{row['MAE']:.2f}")
            with mcol4:
                st.metric("MSE", f"{row['MSE']:.2f}")
    
    st.markdown("---")
    st.markdown('<p class="section-header">Enter Your Habits</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📚 Study & Learning")
        study_hours = st.slider("Study Hours/Day", 0.0, 12.0, 3.0, 0.5, key="study1")
        attendance = st.slider("Attendance %", 0.0, 100.0, 80.0, 1.0, key="att1")
        social_media = st.slider("Social Media Hours", 0.0, 10.0, 2.0, 0.5, key="sm1")
        netflix = st.slider("Netflix/Entertainment Hours", 0.0, 8.0, 1.5, 0.5, key="nf1")
    
    with col2:
        st.markdown("#### 🏃 Health & Wellness")
        sleep_hours = st.slider("Sleep Hours/Night", 0.0, 12.0, 7.0, 0.5, key="sleep1")
        mental_health = st.slider("Mental Health (1-10)", 1, 10, 5, key="mh1")
        exercise = st.slider("Exercise Frequency (days/week)", 0, 7, 3, key="ex1")
        diet = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"], index=1, key="diet1")
    
    with col3:
        st.markdown("#### 🎯 Other Factors")
        part_time = st.selectbox("Part-Time Job", ["No", "Yes"], key="ptj1")
        internet = st.selectbox("Internet Quality", ["Poor", "Average", "Good"], index=1, key="int1")
        extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"], key="extra1")
    
    st.markdown("---")
    
    if st.button("🔮 Predict My Score", key="predict1"):
        # Encode categorical variables
        diet_map = {"Poor": 2, "Fair": 0, "Good": 1}
        internet_map = {"Poor": 2, "Average": 0, "Good": 1}
        yes_no_map = {"No": 0, "Yes": 1}
        
        # Prepare features list
        features = [
            "study_hours_per_day", "attendance_percentage", "mental_health_rating",
            "sleep_hours", "part_time_job", "social_media_hours", "netflix_hours",
            "exercise_frequency", "diet_quality", "internet_quality", "extracurricular_participation"
        ]
        
        # Prepare input data
        input_data = np.array([[
            study_hours, attendance, mental_health, sleep_hours,
            yes_no_map[part_time], social_media, netflix, exercise,
            diet_map[diet], internet_map[internet], yes_no_map[extracurricular]
        ]])
        
        # Prepare dataframe for model training if needed
        df_for_model = df.copy()
        categorical_features = ["part_time_job", "diet_quality", "internet_quality", "extracurricular_participation"]
        for col in categorical_features:
            if col in df_for_model.columns:
                df_for_model[col] = pd.factorize(df_for_model[col])[0]
        
        # Get prediction based on selected model
        if selected_model == "🏆 Best Model (Default - Polynomial Regression)":
            # Use Polynomial Regression as best model
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            X = df_for_model[features]
            y = df_for_model['exam_score']
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            input_poly = poly.transform(input_data)
            best_clf = LinearRegression()
            best_clf.fit(X_poly, y)
            prediction = best_clf.predict(input_poly)[0]
            used_model_name = "Polynomial Regression"
        else:
            # Train and use the selected model (5 models only)
            from sklearn.linear_model import LinearRegression
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.preprocessing import PolynomialFeatures
            
            X = df_for_model[features]
            y = df_for_model['exam_score']
            
            if "Linear Regression" in selected_model:
                selected_clf = LinearRegression()
                selected_clf.fit(X, y)
                prediction = selected_clf.predict(input_data)[0]
                used_model_name = "Linear Regression"
            elif "Polynomial" in selected_model:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X)
                input_poly = poly.transform(input_data)
                selected_clf = LinearRegression()
                selected_clf.fit(X_poly, y)
                prediction = selected_clf.predict(input_poly)[0]
                used_model_name = "Polynomial Regression"
            elif "Decision Tree" in selected_model:
                selected_clf = DecisionTreeRegressor(max_depth=5, min_samples_split=2, random_state=42)
                selected_clf.fit(X, y)
                prediction = selected_clf.predict(input_data)[0]
                used_model_name = "Decision Tree"
            elif "Random Forest" in selected_model:
                selected_clf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
                selected_clf.fit(X, y)
                prediction = selected_clf.predict(input_data)[0]
                used_model_name = "Random Forest"
            elif "Gradient Boosting" in selected_model:
                selected_clf = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
                selected_clf.fit(X, y)
                prediction = selected_clf.predict(input_data)[0]
                used_model_name = "Gradient Boosting"
            else:
                prediction = model.predict(input_data)[0]
                used_model_name = "Default Model"
        
        prediction = max(0, min(100, prediction))
        
        # Display result
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="result-box">
                <p class="score-label">Your Predicted Exam Score (using {used_model_name})</p>
                <p class="score-value">{prediction:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Grade interpretation
            if prediction >= 90:
                grade = "A+"
                color = "#00ff88"
                message = "Outstanding!"
            elif prediction >= 80:
                grade = "A"
                color = "#00f5ff"
                message = "Excellent!"
            elif prediction >= 70:
                grade = "B"
                color = "#88ff00"
                message = "Good Job!"
            elif prediction >= 60:
                grade = "C"
                color = "#ffff00"
                message = "Keep Trying!"
            elif prediction >= 50:
                grade = "D"
                color = "#ff8800"
                message = "Need Improvement"
            else:
                grade = "F"
                color = "#ff0044"
                message = "Critical!"
            
            st.markdown(f"""
            <div style="text-align:center; padding:20px;">
                <p style="font-size:4rem; color:{color}; font-weight:bold;">{grade}</p>
                <p style="color:{color};">{message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Personalized Recommendations
        st.markdown('<p class="section-header">💡 Personalized Recommendations</p>', unsafe_allow_html=True)
        
        recommendations = []
        
        if study_hours < 4:
            potential_gain = (4 - study_hours) * 5
            recommendations.append(f"📚 Increase study hours to 4h/day → Potential +{potential_gain:.0f} points")
        
        if attendance < 85:
            potential_gain = (85 - attendance) * 0.3
            recommendations.append(f"✅ Improve attendance to 85% → Potential +{potential_gain:.0f} points")
        
        if sleep_hours < 7:
            recommendations.append("😴 Get at least 7 hours of sleep for better cognitive function")
        
        if social_media > 3:
            time_saved = social_media - 2
            recommendations.append(f"📱 Reduce social media to 2h/day → Save {time_saved:.1f}h for studying")
        
        if mental_health < 5:
            recommendations.append("🧘 Consider stress management techniques to improve mental health")
        
        if exercise < 3:
            recommendations.append("🏃 Exercise at least 3 times/week to boost energy and focus")
        
        if netflix > 2:
            recommendations.append("📺 Limit entertainment time to focus more on academics")
        
        if not recommendations:
            recommendations.append("🌟 Great habits! Maintain your current routine for continued success!")
        
        for i, rec in enumerate(recommendations[:5]):
            st.info(rec)
        
        if prediction >= 80:
            st.balloons()

# ============== TAB 2: COMPARISON ==============
with tab2:
    st.markdown('<p class="section-header">Compare Two Different Scenarios</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📘 Scenario A (Current)")
        study_a = st.slider("Study Hours", 0.0, 12.0, 2.0, key="sa")
        attendance_a = st.slider("Attendance %", 0.0, 100.0, 70.0, key="aa")
        sleep_a = st.slider("Sleep Hours", 0.0, 12.0, 6.0, key="sla")
        mental_a = st.slider("Mental Health", 1, 10, 4, key="ma")
        ptj_a = st.selectbox("Part-Time Job", ["No", "Yes"], key="pa")
    
    with col2:
        st.markdown("### 📗 Scenario B (Improved)")
        study_b = st.slider("Study Hours", 0.0, 12.0, 5.0, key="sb")
        attendance_b = st.slider("Attendance %", 0.0, 100.0, 90.0, key="ab")
        sleep_b = st.slider("Sleep Hours", 0.0, 12.0, 8.0, key="slb")
        mental_b = st.slider("Mental Health", 1, 10, 7, key="mb")
        ptj_b = st.selectbox("Part-Time Job", ["No", "Yes"], index=0, key="pb")
    
    if st.button("⚖️ Compare Scenarios", key="compare"):
        ptj_map = {"No": 0, "Yes": 1}
        
        try:
            # Try expanded features
            input_a = np.array([[study_a, attendance_a, mental_a, sleep_a, ptj_map[ptj_a], 2.0, 1.5, 3, 0, 0, 0]])
            input_b = np.array([[study_b, attendance_b, mental_b, sleep_b, ptj_map[ptj_b], 2.0, 1.5, 3, 0, 0, 0]])
            pred_a = model.predict(input_a)[0]
            pred_b = model.predict(input_b)[0]
        except:
            input_a = np.array([[study_a, attendance_a, mental_a, sleep_a, ptj_map[ptj_a]]])
            input_b = np.array([[study_b, attendance_b, mental_b, sleep_b, ptj_map[ptj_b]]])
            pred_a = model.predict(input_a)[0]
            pred_b = model.predict(input_b)[0]
        
        pred_a = max(0, min(100, pred_a))
        pred_b = max(0, min(100, pred_b))
        diff = pred_b - pred_a
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="result-box">
                <p class="score-label">Scenario A</p>
                <p class="score-value">{pred_a:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#00ff88" if diff > 0 else "#ff4444"
            arrow = "↑" if diff > 0 else "↓"
            st.markdown(f"""
            <div style="text-align:center; padding:40px;">
                <p style="font-size:3rem; color:{color};">{arrow} {abs(diff):.1f}</p>
                <p style="color:#a0a0a0;">Point Difference</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="compare-box">
                <p class="score-label">Scenario B</p>
                <p class="score-value" style="background: linear-gradient(90deg, #ff00ff, #00f5ff); -webkit-background-clip: text;">{pred_b:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        if diff > 0:
            st.success(f"🎉 Great news! Scenario B scores {diff:.1f} points higher!")
        elif diff < 0:
            st.warning(f"⚠️ Scenario B scores {abs(diff):.1f} points lower")
        else:
            st.info("Both scenarios have similar predicted scores")

# ============== TAB 3: DATA INSIGHTS ==============
with tab3:
    st.markdown('<p class="section-header">📈 Dataset Insights & Statistics</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Avg Exam Score", f"{df['exam_score'].mean():.1f}")
    with col3:
        st.metric("Highest Score", f"{df['exam_score'].max():.1f}")
    with col4:
        st.metric("Lowest Score", f"{df['exam_score'].min():.1f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Score Distribution")
        import plotly.express as px
        fig = px.histogram(df, x='exam_score', nbins=30, 
                          color_discrete_sequence=['#00d4ff'],
                          template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(10, 40, 60, 0.8)',
            plot_bgcolor='rgba(15, 50, 70, 0.6)',
            font_color='#ffffff',
            xaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            yaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Study Hours vs Score")
        fig = px.scatter(df, x='study_hours_per_day', y='exam_score',
                        color='exam_score', color_continuous_scale='viridis',
                        template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(10, 40, 60, 0.8)',
            plot_bgcolor='rgba(15, 50, 70, 0.6)',
            font_color='#ffffff',
            xaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            yaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            coloraxis_colorbar=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation insights
    st.markdown("#### 🔗 Key Correlations with Exam Score")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['exam_score'].drop('exam_score').sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Positive Impact:**")
        for col, corr in correlations.head(4).items():
            st.write(f"• {col.replace('_', ' ').title()}: +{corr:.3f}")
    with col2:
        st.markdown("**📉 Negative Impact:**")
        for col, corr in correlations.tail(4).items():
            st.write(f"• {col.replace('_', ' ').title()}: {corr:.3f}")

# ============== TAB 4: FEATURE ANALYSIS ==============
with tab4:
    st.markdown('<p class="section-header">🔍 What Affects Your Score Most?</p>', unsafe_allow_html=True)
    
    if feature_importance is not None and len(feature_importance) > 0:
        import plotly.express as px
        
        fig = px.bar(feature_importance.sort_values('importance', ascending=True), 
                    x='importance', y='feature', orientation='h',
                    color='importance', color_continuous_scale=[[0, '#ff4444'], [0.5, '#00d4ff'], [1, '#00ffff']],
                    template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(10, 40, 60, 0.8)',
            plot_bgcolor='rgba(15, 50, 70, 0.6)',
            font_color='#ffffff',
            height=500,
            title=dict(text="Feature Importance Analysis", font=dict(color='#ffffff', size=18)),
            xaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            yaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            coloraxis_colorbar=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top insights
        top_features = feature_importance.nlargest(3, 'importance')
        st.markdown("### 🌟 Top 3 Most Important Factors:")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            st.info(f"{i}. **{row['feature'].replace('_', ' ').title()}** - Importance: {row['importance']:.3f}")
    else:
        # Show correlation-based importance
        st.markdown("### 📊 Factor Importance (Based on Correlation)")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'exam_score' in numeric_cols:
            numeric_cols.remove('exam_score')
        if 'student_id' in numeric_cols:
            numeric_cols.remove('student_id')
        
        importance_data = []
        for col in numeric_cols:
            corr = abs(df[col].corr(df['exam_score']))
            importance_data.append({'feature': col, 'importance': corr})
        
        importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
        
        import plotly.express as px
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    color='importance', color_continuous_scale=[[0, '#ff4444'], [0.5, '#00d4ff'], [1, '#00ffff']],
                    template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(10, 40, 60, 0.8)',
            plot_bgcolor='rgba(15, 50, 70, 0.6)',
            font_color='#ffffff',
            height=400,
            xaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            yaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            coloraxis_colorbar=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
        )
        st.plotly_chart(fig, use_container_width=True)

# ============== TAB 5: MODEL COMPARISON ==============
with tab5:
    st.markdown('<p class="section-header">📈 Model Performance Comparison</p>', unsafe_allow_html=True)
    
    if model_results is not None and len(model_results) > 0:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        st.markdown("### 🏆 Compare All Trained Models")
        st.markdown("See how different machine learning models perform on this dataset.")
        
        # Clean model names - remove numbers, unit info, and advanced labels
        model_results_clean = model_results.copy()
        import re
        model_results_clean['model'] = model_results_clean['model'].apply(
            lambda x: re.sub(r'^\d+\.\s*', '', str(x))  # Remove leading numbers
        ).apply(
            lambda x: re.sub(r'\s*\(Unit\s*[IVX]+.*?\)', '', x, flags=re.IGNORECASE)  # Remove (Unit II) etc
        ).apply(
            lambda x: re.sub(r'\s*\(Advanced\)', '', x, flags=re.IGNORECASE)  # Remove (Advanced)
        ).apply(
            lambda x: re.sub(r'\s*-\s*Lazy Learning', '', x, flags=re.IGNORECASE)  # Remove - Lazy Learning
        ).str.strip()
        
        # Display model results table - select only key columns
        st.markdown("#### 📋 Performance Metrics Table")
        display_cols = ['model', 'MAE', 'MSE', 'RMSE', 'R2']
        if all(col in model_results_clean.columns for col in display_cols):
            styled_results = model_results_clean[display_cols].copy()
            styled_results.columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R² Score']
            st.dataframe(styled_results.style.highlight_max(subset=['R² Score'], color='#00d4ff30')
                        .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='#00d4ff30')
                        .format({'MAE': '{:.4f}', 'MSE': '{:.4f}', 'RMSE': '{:.4f}', 'R² Score': '{:.4f}'}),
                        use_container_width=True)
        else:
            st.dataframe(model_results_clean, use_container_width=True)
        
        st.markdown("---")
        
        # R² Score Comparison Bar Chart
        st.markdown("#### 🎯 R² Score Comparison (Higher is Better)")
        fig_r2 = px.bar(model_results_clean.sort_values('R2', ascending=True), 
                       x='R2', y='model', orientation='h',
                       color='R2', color_continuous_scale=[[0, '#ff4444'], [0.5, '#00d4ff'], [1, '#00ffff']],
                       template='plotly_dark',
                       labels={'R2': 'R² Score', 'model': 'Model'})
        fig_r2.update_layout(
            paper_bgcolor='rgba(10, 40, 60, 0.8)',
            plot_bgcolor='rgba(15, 50, 70, 0.6)',
            font_color='#ffffff',
            height=400,
            showlegend=False,
            xaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            yaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            coloraxis_colorbar=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
        )
        fig_r2.add_vline(x=0.9, line_dash="dash", line_color="#ff00ff", 
                        annotation_text="90% Threshold", annotation_position="top")
        st.plotly_chart(fig_r2, use_container_width=True)
        
        st.markdown("---")
        
        # RMSE Comparison
        st.markdown("#### 📉 RMSE Comparison (Lower is Better)")
        fig_rmse = px.bar(model_results_clean.sort_values('RMSE', ascending=False), 
                         x='RMSE', y='model', orientation='h',
                         color='RMSE', color_continuous_scale=[[0, '#00ffff'], [0.5, '#00d4ff'], [1, '#ff4444']],
                         template='plotly_dark',
                         labels={'RMSE': 'Root Mean Square Error', 'model': 'Model'})
        fig_rmse.update_layout(
            paper_bgcolor='rgba(10, 40, 60, 0.8)',
            plot_bgcolor='rgba(15, 50, 70, 0.6)',
            font_color='#ffffff',
            height=400,
            showlegend=False,
            xaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            yaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
            coloraxis_colorbar=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
        
        st.markdown("---")
        
        # Combined Metrics Radar/Spider Chart
        st.markdown("#### 🕸️ Multi-Metric Comparison")
        
        # Normalize metrics for radar chart (0-1 scale)
        metrics_normalized = model_results_clean.copy()
        metrics_normalized['R2_norm'] = metrics_normalized['R2']
        metrics_normalized['MAE_norm'] = 1 - (metrics_normalized['MAE'] / metrics_normalized['MAE'].max())
        metrics_normalized['RMSE_norm'] = 1 - (metrics_normalized['RMSE'] / metrics_normalized['RMSE'].max())
        
        # Create subplot with all metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**MAE Comparison (Lower is Better)**")
            fig_mae = px.bar(model_results_clean.sort_values('MAE', ascending=False), 
                           x='MAE', y='model', orientation='h',
                           color='MAE', color_continuous_scale=[[0, '#00ffff'], [0.5, '#00d4ff'], [1, '#ff4444']],
                           template='plotly_dark')
            fig_mae.update_layout(
                paper_bgcolor='rgba(10, 40, 60, 0.8)',
                plot_bgcolor='rgba(15, 50, 70, 0.6)',
                font_color='#ffffff',
                height=350,
                showlegend=False,
                xaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
                yaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
                coloraxis_colorbar=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col2:
            st.markdown("**MSE Comparison (Lower is Better)**")
            fig_mse = px.bar(model_results_clean.sort_values('MSE', ascending=False), 
                           x='MSE', y='model', orientation='h',
                           color='MSE', color_continuous_scale=[[0, '#00ffff'], [0.5, '#00d4ff'], [1, '#ff4444']],
                           template='plotly_dark')
            fig_mse.update_layout(
                paper_bgcolor='rgba(10, 40, 60, 0.8)',
                plot_bgcolor='rgba(15, 50, 70, 0.6)',
                font_color='#ffffff',
                height=350,
                showlegend=False,
                xaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
                yaxis=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')),
                coloraxis_colorbar=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
            )
            st.plotly_chart(fig_mse, use_container_width=True)
        
        st.markdown("---")
        
        # Best Model Highlight
        best_model_name = model_results_clean.loc[model_results_clean['R2'].idxmax(), 'model']
        best_r2 = model_results_clean['R2'].max()
        best_rmse = model_results_clean.loc[model_results_clean['R2'].idxmax(), 'RMSE']
        
        st.markdown("### 🏆 Best Performing Model")
        st.success(f"""
        **{best_model_name}** achieves the highest R² score of **{best_r2:.4f}** ({best_r2*100:.2f}%)
        
        - 📊 This model explains {best_r2*100:.1f}% of the variance in exam scores
        - 📉 Average prediction error (RMSE): {best_rmse:.2f} points
        """)
        
        # Model Categories
        st.markdown("### 📚 Model Categories")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(255,0,255,0.1)); padding: 20px; border-radius: 15px; border: 1px solid #00f5ff;">
            <h4 style="color: #00f5ff;">📖 Syllabus Models (3)</h4>
            <ul style="color: #e0e0e0;">
            <li>Linear Regression</li>
            <li>Polynomial Regression</li>
            <li>Decision Tree</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(255,0,255,0.1), rgba(0,245,255,0.1)); padding: 20px; border-radius: 15px; border: 1px solid #ff00ff;">
            <h4 style="color: #ff00ff;">🚀 Advanced Models (2)</h4>
            <ul style="color: #e0e0e0;">
            <li>Random Forest</li>
            <li>Gradient Boosting</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.warning("⚠️ Model comparison data not available. Please run the training notebook first.")
        st.info("""
        To see model comparison:
        1. Open `project.ipynb`
        2. Run all cells to train the models
        3. The results will be saved to `model_results.csv`
        4. Refresh this page
        """)

# Footer
st.markdown("---")
