# healthcare-insurance-charges
This project explores predicting medical insurance charges using machine learning. The workflow included thorough data cleaning, handling missing values, encoding categorical variables, and addressing outliers before training and optimizing models with hyperparameter tuning. Model performance was then evaluated on unseen data to assess effectiveness.

Project is adapted from DataCamp: https://app.datacamp.com/learn/projects/2264

## How it's made
Language: Python 

Packages: pandas, scikit-learn, seaborn and matplotlib

### Data Preparation
- Loaded the dataset with **pandas** and explored data types and distributions.  
- Cleaned the target variable `charges` (removed `$` signs, converted to float).  
- Standardized categorical values (`sex` and `region`) for consistency.  
- Corrected negative values in numerical columns (`age`, `children`, `bmi`) by taking absolute values.  
- Identified outliers using the IQR method: BMI outliers were kept due to low frequency, and no transformations(log or sqrt) were applied to `charges` to avoid unrealistic predictions.  
- Removed duplicate rows.  

### Handling Missing Data
- Visualized missing values with a **seaborn** heatmap.  
- Dropped rows with missing `charges` values to avoid bias in the target variable.  
- For other missing values:  
  - Used **Iterative Imputation** for numerical featuresfrom **scikit-learn** that creates a value based on other features.  
  - Used **Simple Imputation** for categorical features from **scikit-learn** that takes the most-frequent value.  

### Feature Engineering
- One-Hot Encoded categorical features with **pandas**.  
- Split dataset into training (80%) and testing (20%) sets.  

### Modeling
- Built a machine learning pipeline with **StandardScaler** and **Linear Regression** using **scikit-learn**.  
- Performed hyperparameter tuning with **GridSearchCV** (cross-validation) from **scikit-learn**.  
- Evaluated models using **R²** and **RMSE**, and selected the best-performing configuration.  

### Validation
- Loaded an external validation dataset.  
- Generated predictions for `charges` using the trained model.  
- Applied post-processing by capping predictions to be greater than 1000.  

## Future Directions
- **Model experimentation**: Try alternative algorithms such as Random Forest, Gradient Boosting, or XGBoost to compare performance with Linear Regression.
- **Outlier treatment**: Explore models less sensitive to extreme values or experiment with alternative transformations beyond log and square root.

## Lessons Learned
- The importance of thorough **data cleaning** before modeling — inconsistent categories, missing values, and outliers had a significant impact on results.  
- **Feature scaling and encoding** are essential for linear models to ensure fair treatment of variables.  
- Overly aggressive transformations (e.g., log/sqrt on `charges`) can sometimes make results less realistic, highlighting the trade-off between statistical performance and practical interpretability.  
- Iterative Imputation provided a more robust way of handling missing numerical values compared to simple imputation.  
- Cross-validation and hyperparameter tuning with **GridSearchCV** improved model reliability compared to using default parameters.  

