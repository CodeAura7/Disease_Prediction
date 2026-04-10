# 🩺 Disease Prediction System using Machine Learning
### GTU Internship Project | Healthcare Domain

---

## 📌 Problem Statement
Diabetes is a chronic disease affecting millions globally. Early detection is
crucial but requires clinical expertise. This project builds a Machine Learning
system that predicts diabetes risk from 8 simple clinical measurements —
making early screening accessible and data-driven.

---

## 📂 Project Folder Structure
```
disease_prediction/
│
├── train_model.py          # ← Main ML pipeline (EDA → Train → Evaluate → Save)
├── requirements.txt        # ← Python dependencies
├── viva_qa.py              # ← Viva Q&A reference
├── README.md               # ← This file
│
├── data/
│   └── diabetes.csv        # ← Dataset (download from Kaggle)
│
├── models/
│   ├── best_model.pkl      # ← Saved trained model (generated)
│   ├── scaler.pkl          # ← Saved StandardScaler (generated)
│   ├── eda_distributions.png
│   ├── eda_correlation.png
│   ├── eda_outcome.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
└── app/
    └── app.py              # ← Streamlit web application
```

---

## 📦 Dataset
**Pima Indians Diabetes Dataset**
- Source : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Size   : 768 samples, 8 features + 1 target
- Target : Outcome (0 = no diabetes, 1 = diabetes)

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (μU/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train_model.py
```
This will:
- Perform EDA and save charts to `models/`
- Train 3 ML models and compare them
- Save the best model as `models/best_model.pkl`

### Step 3 — Launch the web app
```bash
streamlit run app/app.py
```
Open your browser at: http://localhost:8501

---

## 🤖 Models Trained
| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Linear | Fast, interpretable baseline |
| Random Forest | Ensemble | Handles non-linearity, gives feature importance |
| SVM (RBF Kernel) | Kernel-based | Effective in high-dimensional space |

---

## 📊 Evaluation Metrics Used
- **Accuracy** — Overall correct predictions
- **Precision** — Of predicted diabetics, how many actually are
- **Recall** — Of actual diabetics, how many did we catch
- **F1-Score** — Balance of precision and recall
- **Confusion Matrix** — Visual breakdown of prediction errors
- **Cross-Validation** — 5-fold CV for robust performance estimate

---

## 💡 Future Improvements
1. SMOTE to handle class imbalance
2. GridSearchCV / RandomizedSearchCV for hyperparameter tuning
3. Add XGBoost, LightGBM, or Neural Network
4. SHAP values for explainable AI
5. Deploy on Streamlit Cloud or Heroku
6. Build a REST API with Flask/FastAPI
7. Add multi-disease prediction module
8. Real-time data from electronic health records

---

## 🎓 Resume Points
- Developed an end-to-end ML pipeline for diabetes prediction achieving X% accuracy
- Applied EDA, preprocessing, and feature engineering on medical datasets
- Trained and compared Logistic Regression, Random Forest, and SVM classifiers
- Built an interactive Streamlit web app for real-time clinical decision support
- Deployed model using Pickle serialisation for production-ready inference

---

*GTU Internship · Healthcare AI · Python ML Project*
