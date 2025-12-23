# Customer-Churn-Prediction
Machine Learning mini-project predicting customer churn using Online Retail data.

# 📉 Customer Churn Prediction with RFM Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Business Goal:** Minimize revenue loss by identifying "silent attrition" and targeting high-value customers at risk of churning.

## 📖 Project Overview

Customer acquisition is expensive; retention is key to profitability. This project builds a predictive system to identify customers who have likely churned (defined as **90 days of inactivity**) based on their purchasing history.

By transforming raw transactional data into **RFM (Recency, Frequency, Monetary)** features, we trained machine learning models to classify customers as "Active" or "Churned" with high precision, allowing the business to proactively launch win-back campaigns.

## 🔍 Key Findings & Insights

* **Churn Definition:** A customer is considered churned if they have not made a purchase in the last **90 days**.
* **Class Imbalance:** The dataset consists of approximately **66% Active** and **33% Churned** customers.
* **Top Predictor:** **Recency** (days since last purchase) was the strongest predictor, showing a correlation of **+0.82** with churn.
* **Model Accuracy:** The final models achieved near-perfect scores on the test set, validating the 90-day window as a definitive separator for this specific customer base.

## 🛠️ Methodology

The project follows a standard Data Science lifecycle:

1.  **Data Preprocessing**:
    * Aggregated transactional data to the customer level.
    * Calculated **Recency**, **Frequency**, and **Monetary** values for each customer.
2.  **Exploratory Data Analysis (EDA)**:
    * Visualized distributions of RFM features.
    * Identified the strong linear relationship between Recency and Churn.
3.  **Feature Engineering**:
    * Scaled features using `StandardScaler` to normalize distributions.
    * Addressed class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) within an `ImbPipeline`.
4.  **Model Training**:
    * **Logistic Regression**: Tuned with `GridSearchCV` (Optimal `C=100`).
    * **Random Forest**: Trained as a comparative ensemble model.

## 📊 Performance Metrics

Both models demonstrated exceptional performance in distinguishing between active and churned customers:

| Model | F1-Score | AUC Score | Cross-Val F1 |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **1.00** | **1.00** | **0.999** |
| **Random Forest** | **1.00** | **1.00** | N/A |

*> **Note:** The perfect scores suggest that "Recency" is a deterministic feature for the chosen churn definition (90 days).*

## 💡 Actionable Strategy (Business Value)

The technical analysis translates directly into business action:

* **Target List:** We generated a prioritized list of the **Top 20 Churned Customers by Monetary Value**.
* **Recommendation:** These high-value individuals should be the immediate focus of a personalized marketing "Win-Back" campaign (e.g., exclusive discounts, "We Miss You" emails).
* **Impact:** Focusing resources on these specific users prevents the highest potential revenue leakage.

## 💻 Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (LogisticRegression, RandomForest, GridSearchCV, Pipeline)
* **Imbalanced Learning:** Imbalanced-learn (SMOTE)
* **Visualization:** Matplotlib, Seaborn

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* Jupyter Notebook

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/customer-churn-rfm.git](https://github.com/yourusername/customer-churn-rfm.git)
    cd customer-churn-rfm
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
    ```

3.  **Run the Notebook:**
    ```bash
    jupyter notebook "Customer_Churn_Model.ipynb"
    ```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improving the churn definition or modeling approach.

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
