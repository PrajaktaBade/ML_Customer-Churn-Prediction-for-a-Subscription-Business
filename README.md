## 📁 Customer Churn Prediction for a Subscription Business
### Project Overview
This project predicts customer churn in a subscription-based business. Churn prediction helps identify customers likely to leave, enabling proactive retention strategies to increase customer lifetime value and reduce revenue loss.

### Dataset
- Source: WA_Fn-UseC_-Telco-Customer-Churn.csv
- Records: 7043 customers
- Features: Customer demographics, account info, billing, and contract details
- Target: Churn (Yes/No)

**Key Insights from Dataset:**

- Majority of customers (~73%) do not churn → class imbalance exists.
- Features like tenure, MonthlyCharges, and contract type correlate with churn.
- TotalCharges contains missing or invalid values that require cleaning.

### Project Workflow
**Phase 1. Data Cleaning & Preprocessing**
- Converted TotalCharges to numeric and filled missing values with median.
- Encoded target column (Churn) as 1=Yes, 0=No.
- One-hot encoded categorical variables using pd.get_dummies.
- Scaled numeric features (tenure, MonthlyCharges, TotalCharges) with StandardScaler.

**Phase 2. Train-Test Split**
- Stratified 80/20 train-test split to preserve class distribution.

**Phase 3. Baseline Model: Logistic Regression**
- Default Logistic Regression achieved accuracy ~0.80.
- Recall for churners (class 1) was low (~0.55), indicating the model missed many at-risk customers.

**Phase 4: Random Forest**
- Random Forest with class_weight='balanced' did not improve recall (~0.50).
- Accuracy remained ~0.80.
- Baseline Logistic Regression performed better for minority class prediction.

**Phase 5: Handling Class Imbalance**
- Logistic Regression with class_weight='balanced' improved recall for churners to ~0.75.
- Accuracy slightly dropped to 0.75 — expected trade-off.
- Confusion matrix shows reduced false negatives → more churners correctly identified.

### Key Takeaways
- Handling class imbalance is critical in churn prediction.
- Prioritizing recall for churners aligns better with business goals than accuracy.
- Logistic Regression with class weighting provides a simple, interpretable, and effective model.
- Random Forest can be explored in future for more complex feature interactions.

### Future Improvements
- Tune decision threshold (<0.5) to further improve recall.
- Explore ensemble methods like Random Forest, XGBoost, or LightGBM with hyperparameter tuning.
- Feature importance analysis to guide targeted business actions for churn prevention.
- Deploy model through an API for real-time churn alerts.

### Project Structure
```
ML_Customer-Churn-Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_prediction.ipynb
├── README.md
```
### Tools & Libraries
- Python 3.x
- Pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, preprocessing, metrics)
- StandardScaler for numeric feature scaling
