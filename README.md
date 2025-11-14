# ShipmentSure: On-Time Delivery Prediction using Machine Learning

## Project Overview
The On-Time Delivery Prediction project focuses on predicting whether a shipment will be delivered on time or delayed based on logistics, customer behavior, and product-related attributes. This project supports supply chain optimization and enhances customer satisfaction through data-driven insights.

## Problem Statement
Predicting shipment delivery status is challenging due to various influencing factors such as transportation mode, customer interaction, product characteristics, and more. The goal is to build an accurate classification model to proactively detect potential delivery delays.

## Objectives
- Clean and preprocess logistics data  
- Engineer meaningful features  
- Encode categorical variables  
- Scale numerical features  
- Handle class imbalance  
- Train and evaluate multiple ML models  
- Identify the best-performing model  

## Dataset
The dataset (`Train.csv`) contains shipment-level details.

### Key Features:
- Warehouse_block  
- Mode_of_Shipment  
- Customer_care_calls  
- Customer_rating  
- Cost_of_the_Product  
- Prior_purchases  
- Product_importance  
- Gender  
- Discount_offered  
- Weight_in_gms  
- Reached.on.Time_Y.N (Target)

## Technologies Used
- Python (3.13.4)
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost, LightGBM, CatBoost
- joblib
- VS Code

### Library Versions Used:
```
jupyter==1.1.1
matplotlib==3.10.6
numpy==2.3.3
pandas==2.3.3
scikit-learn==1.7.2
seaborn==0.13.2
streamlit==1.51.0
joblib==1.3.2
```

## Data Preprocessing
- Checked for duplicates and missing values  
- Label Encoding for ordinal categories  
- One-Hot Encoding for nominal categories  
- StandardScaler applied to numerical variables  
- Engineered new ratio feature: Cost_to_Weight_ratio  
- SMOTE used for class balancing  

## Model Building
Models trained:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Naive Bayes  
- KNN  
- SVM  
- XGBoost  
- LightGBM  
- CatBoost  

Evaluation done using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- 10-fold Stratified Cross-Validation  

## Best Performing Model
- **LightGBM** and **XGBoost** achieved the best overall performance with strong ROC-AUC and balanced precision-recall.  
- CatBoost also performed competitively.


## Deployment
Model can be deployed using:
- Streamlit Web App   

## Author
**Sruthi A S**  
GitHub: https://github.com/Sruthi860
