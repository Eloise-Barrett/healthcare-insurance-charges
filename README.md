# healthcare-insurance-charges
Using machine learning to predict the cost of medical care billed by insurance. The project required extensive data cleaning, hyperparameter tuning and evaluating the effectiveness of the model produced on unseen data.

Project is from DataCamp: https://app.datacamp.com/learn/projects/2264

## How it's made
Language: Python 

Packages: pandas, scikit-learn, seaborn and matplotlib

### Data Preparation
- Loaded the dataset with **pandas** and explored data types and distributions.  
- Cleaned the target variable `charges` (removed `$` signs, converted to float).  
- Standardized categorical values (`sex` and `region`) for consistency.  
- Corrected negative values in numerical columns (`age`, `children`, `bmi`) by taking absolute values.  
- Identified outliers using the IQR method: BMI outliers were kept due to low frequency, and no transformations were applied to `charges` to avoid unrealistic predictions.  
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

