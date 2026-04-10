# 
# VIVA QUESTIONS & ANSWERS
# Disease Prediction System using Machine Learning
# GTU Internship — Healthcare Domain
# 

VIVA_QA = """
╔══════════════════════════════════════════════════════════════╗
║         VIVA QUESTIONS & ANSWERS — DISEASE PREDICTION       ║
╚══════════════════════════════════════════════════════════════╝


SECTION A : BASICS


Q1. What is the problem you are solving in this project?
A.  We predict whether a patient is at risk of diabetes based on
    clinical measurements like glucose, BMI, blood pressure, age,
    etc. This is a binary classification problem — the output is
    either 0 (no diabetes) or 1 (diabetes).

Q2. Which dataset did you use and why?
A.  Pima Indians Diabetes Dataset from the UCI Machine Learning
    Repository / Kaggle. It has 768 patients with 8 clinical
    features, making it ideal for binary classification. It's
    widely used in ML healthcare benchmarks.

Q3. What is Machine Learning?
A.  Machine Learning is a branch of AI where computers learn
    patterns from data without being explicitly programmed. We
    train a model on historical data so it can predict outcomes
    on new, unseen data.

Q4. What is the difference between supervised and unsupervised learning?
A.  Supervised: training data has labels (e.g., diabetic / not).
    We used supervised learning here.
    Unsupervised: no labels — model finds patterns (clustering,
    dimensionality reduction).


SECTION B : DATA & PREPROCESSING


Q5. What is EDA and why is it important?
A.  Exploratory Data Analysis (EDA) is the process of analysing
    data using statistics and visualisations before modelling.
    It helps us understand distributions, spot outliers,
    identify correlations, and uncover missing values.

Q6. How did you handle missing values in this dataset?
A.  Features like Glucose, BMI, and BloodPressure had zero values
    which are medically impossible. We treated these as missing
    values and replaced them with the column median (robust to
    outliers).

Q7. What is feature scaling? Which method did you use?
A.  Feature scaling normalises the range of features so no single
    feature dominates due to its magnitude.
    We used StandardScaler which transforms each feature to have
    mean=0 and standard deviation=1 (Z-score normalisation).
    Formula: z = (x − μ) / σ

Q8. Why do we split data into train and test sets?
A.  To evaluate how well the model generalises to unseen data.
    If we tested on training data, the model would appear perfect
    but fail in real-world use (overfitting). We used an 80/20
    train/test split.


SECTION C : ML MODELS


Q9. Explain Logistic Regression.
A.  Despite its name, it's a classification algorithm. It models
    the probability of an outcome using the sigmoid function
    which maps any value to [0,1]. If probability > 0.5 → class 1.
    It's interpretable and works well on linearly separable data.

Q10. Explain Random Forest.
A.   An ensemble learning method that builds multiple decision
     trees on random subsets of data (bagging) and combines their
     predictions by majority voting. It reduces overfitting and
     handles non-linear relationships. The feature_importances_
     attribute tells us which features matter most.

Q11. Explain SVM (Support Vector Machine).
A.   SVM finds the optimal hyperplane that best separates the two
     classes with maximum margin. Support vectors are the data
     points closest to the hyperplane. The RBF kernel handles
     non-linear data by mapping it to higher dimensions.

Q12. What is Cross-Validation?
A.   A technique to assess model performance reliably. In k-fold
     CV (k=5), we split training data into 5 parts, train on 4
     and validate on 1 — repeated 5 times. The average score is
     more reliable than a single train/test split.


SECTION D : EVALUATION


Q13. What is a Confusion Matrix?
A.   A 2×2 table showing model predictions vs actual labels:
       ┌────────────────┬──────────────┬───────────────┐
       │                │ Predicted: 0 │ Predicted: 1  │
       ├────────────────┼──────────────┼───────────────┤
       │  Actual: 0     │ True Neg(TN) │ False Pos(FP) │
       │  Actual: 1     │ False Neg(FN)│ True Pos(TP)  │
       └────────────────┴──────────────┴───────────────┘

Q14. Define Precision, Recall, and F1-Score.
A.   Precision = TP / (TP + FP)  — Of all predicted positives,
                                    how many are actually positive?
     Recall    = TP / (TP + FN)  — Of all actual positives,
                                    how many did we catch?
     F1-Score  = 2 × (P × R) / (P + R)  — Harmonic mean of both.

     In healthcare, Recall (sensitivity) is more important —
     missing a diabetic patient (FN) is more dangerous than
     a false alarm (FP).

Q15. Why not just use accuracy as the metric?
A.   Accuracy is misleading on imbalanced datasets. If 90% of
     patients are healthy, a model that always predicts "no
     disease" gets 90% accuracy but is useless medically.
     Precision and recall give a fuller picture.


SECTION E : DEPLOYMENT & TOOLS


Q16. What is Pickle and why did you use it?
A.   Pickle is Python's built-in serialisation module. We use it
     to save the trained ML model and scaler to disk (.pkl files)
     so they can be loaded later by the web app without retraining.

Q17. What is Streamlit?
A.   Streamlit is an open-source Python library for building
     interactive web applications for data/ML projects without
     writing HTML, CSS, or JavaScript. We used it to build the
     prediction frontend.

Q18. Why did you save the scaler separately?
A.   The scaler was fitted on training data. When the app receives
     new patient inputs, they must be scaled using the SAME
     parameters (mean, std) the model was trained on. Saving the
     scaler ensures consistent preprocessing at prediction time.

Q19. What is overfitting? How did you prevent it?
A.   Overfitting is when a model memorises training data and
     performs poorly on new data. We prevented it by:
     - Using cross-validation to detect it
     - Using Random Forest (bagging reduces variance)
     - Feature scaling for SVM/LR
     - Train/test split for unbiased evaluation

Q20. How would you improve this project further?
A.   1. Use SMOTE to handle class imbalance
     2. Hyperparameter tuning with GridSearchCV
     3. Add XGBoost/Neural Network for better accuracy
     4. Deploy on cloud (Heroku / AWS / Streamlit Cloud)
     5. Add SHAP values for model explainability
     6. Build REST API with Flask/FastAPI
     7. Use real-time patient data from hospital APIs
     8. Add multi-disease prediction (heart disease, cancer)


SECTION F : BONUS / TRICKY QUESTIONS


Q21. What is the bias-variance tradeoff?
A.   Bias: error from wrong assumptions (underfitting).
     Variance: sensitivity to small fluctuations in training data
     (overfitting). Good models balance both for low total error.

Q22. Why did you use RBF kernel in SVM?
A.   The dataset is non-linearly separable. RBF (Radial Basis
     Function) kernel implicitly maps inputs to infinite-
     dimensional space, allowing SVM to find a linear separator
     in that space, effectively creating non-linear boundaries
     in the original feature space.

Q23. What does feature importance in Random Forest represent?
A.   It measures how much each feature decreases impurity
     (Gini index) across all trees. Higher value = more
     important for prediction. In our project, Glucose and BMI
     typically rank highest.

Q24. What is the difference between predict() and predict_proba()?
A.   predict()      → returns class label (0 or 1)
     predict_proba() → returns probability for each class [P(0), P(1)]
     We use predict_proba() to show confidence % in the app.

Q25. If you had to choose one model for production, which would you pick?
A.   Random Forest — because it:
     ✓ Handles non-linear relationships
     ✓ Is robust to outliers
     ✓ Provides feature importance
     ✓ Rarely overfits with proper n_estimators
     ✓ Works well without extensive hyperparameter tuning
"""

if __name__ == "__main__":
    print(VIVA_QA)
