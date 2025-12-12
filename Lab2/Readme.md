# 🏠 Simple Linear Regression - House Price Prediction

**Complete explanation** of the Simple Linear Regression experiment for predicting house sale prices using living area and number of bedrooms.

---

## 📋 **Overview**

This lab demonstrates:
- Loading training and test datasets
- Feature selection and data cleaning
- Training a Linear Regression model
- Model evaluation using validation data
- Making predictions on unseen test data
- Creating comprehensive visualizations

**Dataset:** House Prices dataset with features like GrLivArea (above ground living area) and BedroomAbvGr (bedrooms above ground)

---

# ✅ **STEP 1 — Import Required Packages**

### ✔ What we did:

Imported necessary Python libraries:

* **pandas** → data loading & manipulation
* **numpy** → numeric operations
* **matplotlib.pyplot** → plotting & visualization
* **sklearn.linear_model** → LinearRegression model
* **sklearn.metrics** → mean_squared_error, r2_score for evaluation
* **sklearn.model_selection** → train_test_split for data splitting

### ✔ Why:

These libraries provide the essential tools for:
- Loading and processing CSV files
- Building regression models
- Evaluating model performance
- Creating visualizations

### ✔ What it shows:

Nothing visual — sets up the environment for machine learning workflow.

---

# ✅ **STEP 2 — Load Data**

### ✔ What we did:

Loaded two separate CSV files:

```python
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
```

### ✔ Why:

* **train.csv** contains features + target (SalePrice)
* **test.csv** contains only features (no SalePrice)

This mimics real-world scenarios where we train on labeled data and predict on unlabeled data.

### ✔ What it shows:

* First few rows of training data using `train_df.head()`
* Dataset structure with multiple columns
* Target variable: **SalePrice**

---

# ✅ **STEP 3 — Select Required Columns**

### ✔ What we did:

Selected specific features for modeling:

**From training data:**
* **GrLivArea** → Above ground living area (square feet)
* **BedroomAbvGr** → Number of bedrooms above ground
* **SalePrice** → Target variable (house sale price)

**From test data:**
* **GrLivArea**
* **BedroomAbvGr**
* (No SalePrice — this is what we'll predict)

Removed rows with missing values using `.dropna()`.

### ✔ Why:

* Focus on relevant features that logically impact house prices
* Clean data ensures model training works properly
* Test data lacks the target variable (real prediction scenario)

### ✔ What it shows:

* `train_clean`: 3 columns (2 features + 1 target)
* `test_clean`: 2 columns (2 features only)
* All missing values removed

---

# ✅ **STEP 4 — Create Train/Validation Split**

### ✔ What we did:

Split the training data into two parts:

* **Training set** (80%) → used to train the model
* **Validation set** (20%) → used to evaluate model performance

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Where:
* **X** = features (GrLivArea, BedroomAbvGr)
* **y** = target (SalePrice)

### ✔ Why:

We need to evaluate how well the model performs on **unseen data**:

* Training on 80% of data
* Testing on remaining 20% (model has never seen this)
* This prevents overfitting and measures generalization

`random_state=42` ensures reproducibility.

### ✔ What it shows:

* X_train, y_train → for model training
* X_valid, y_valid → for model evaluation
* Split ratio: 80-20

---

# ✅ **STEP 5 — Train Model**

### ✔ What we did:

Created and trained a **Linear Regression** model:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The model learns the equation:

```
SalePrice = intercept + (coef₁ × GrLivArea) + (coef₂ × BedroomAbvGr)
```

### ✔ Why:

Linear Regression finds the best-fit line that:
* Minimizes prediction errors
* Establishes relationship between features and target
* Provides interpretable coefficients

### ✔ What it shows:

* **Coefficients** → weight of each feature
  * Positive coefficient = feature increases price
  * Negative coefficient = feature decreases price
* **Intercept** → baseline price when all features are zero

Example output:
```
Coefficients: [107.13, -8247.83]
Intercept: 13456.78
```

This means:
* Each sq ft increases price by ~$107
* Each bedroom decreases price by ~$8,248 (holding area constant)

---

# ✅ **STEP 6 — Validate Model (Evaluate)**

### ✔ What we did:

Evaluated model performance on validation data:

```python
y_pred = model.predict(X_valid)
```

Calculated three metrics:

* **MSE (Mean Squared Error)** → average of squared errors
* **R² Score** → proportion of variance explained (0 to 1)
* **RMSE (Root Mean Squared Error)** → square root of MSE

### ✔ Why:

These metrics quantify how well the model predicts:

* **RMSE** tells us typical prediction error in dollars
* **R²** shows how much variance the model captures
  * R² = 0.75 means model explains 75% of price variation
  * R² = 1.0 means perfect predictions

### ✔ What it shows:

Example output:
```
MSE: 1,234,567,890
RMSE: 35,136
R² Score: 0.58
```

* Average prediction error: ~$35,136
* Model explains 58% of price variance
* Lower RMSE and higher R² indicate better performance

---

# ✅ **STEP 7 — Train Final Model on Full Training Data**

### ✔ What we did:

Retrained the model on **entire training dataset** (not just 80%):

```python
final_model = LinearRegression()
final_model.fit(X, y)
```

### ✔ Why:

After validation confirms the model works well:
* Use all available training data for maximum learning
* Better final model for real predictions
* No data is "wasted" on validation anymore

### ✔ What it shows:

A refined model trained on 100% of training data, ready for deployment.

---

# ✅ **STEP 8 — Predict on Real Test Dataset**

### ✔ What we did:

Made predictions on the external test data:

```python
test_predictions = final_model.predict(test_clean)
```

### ✔ Why:

This is the **real application** of the model:
* Test data has no SalePrice
* Model generates price predictions
* Simulates real-world use case

### ✔ What it shows:

Array of predicted prices for each house in test dataset:
```
[254123.45, 189456.78, 312890.12, ...]
```

First 10 predictions displayed for verification.

---

# ✅ **STEP 9 — Save Predictions**

### ✔ What we did:

Created a CSV file with predictions:

```python
output = pd.DataFrame({
    "Id": test_df["Id"], 
    "PredictedSalePrice": test_predictions
})
output.to_csv("predictions.csv", index=False)
```

### ✔ Why:

* Save results for submission or further analysis
* Maintain house IDs for tracking
* Standardized output format

### ✔ What it shows:

**predictions.csv** file with:
* Id column → house identifier
* PredictedSalePrice column → model predictions

---

# 📊 **VISUALIZATION STEPS**

# ✅ **STEP 10 — Regression Line Visualization**

### ✔ What we did:

Plotted **GrLivArea vs SalePrice** with regression line:

* Blue scatter points → actual training data
* Red line → model's predicted relationship

### ✔ Why:

Visualizes how well the linear model fits the data:
* Line through the middle of points → good fit
* Points far from line → prediction errors
* Shows linear relationship assumption

### ✔ What it shows:

A scatter plot with regression line overlay showing the model's learned relationship between living area and price.

---

# ✅ **STEP 11 — Bedrooms vs SalePrice**

### ✔ What we did:

Created scatter plot of **Number of Bedrooms vs SalePrice**.

### ✔ Why:

Explore the second feature's relationship with price:
* Do more bedrooms mean higher prices?
* Is the relationship linear?
* Are there outliers?

### ✔ What it shows:

Scatter plot revealing bedroom-price relationship. Typically shows clusters at discrete bedroom counts (2, 3, 4, etc.).

---

# ✅ **STEP 12 — Color-Coded Scatter Plot**

### ✔ What we did:

Plotted **GrLivArea vs SalePrice** colored by **number of bedrooms**:

* X-axis: GrLivArea
* Y-axis: SalePrice
* Color: BedroomAbvGr (using viridis colormap)

### ✔ Why:

Shows **three dimensions** in a 2D plot:
* How area and price relate
* How bedrooms influence this relationship
* Identifies patterns (e.g., same area, different bedrooms = different prices)

### ✔ What it shows:

Multi-dimensional visualization with color gradient representing bedroom count. Helps understand how both features interact.

---

# ✅ **STEP 13 — 3D Scatter Plot**

### ✔ What we did:

Created a 3D visualization:

* X-axis: GrLivArea
* Y-axis: BedroomAbvGr
* Z-axis: SalePrice
* Color: Price (plasma colormap)

### ✔ Why:

Visualize **all three variables simultaneously**:
* See the 3D "surface" the model tries to fit
* Understand complex relationships
* Identify outliers in 3D space

### ✔ What it shows:

Interactive 3D scatter plot showing the true multidimensional nature of the data. Higher prices typically appear at higher areas.

---

# ✅ **STEP 14 — Residual Plot**

### ✔ What we did:

Plotted **residuals** (errors) vs predicted values:

* X-axis: Predicted SalePrice
* Y-axis: Residuals (Actual - Predicted)
* Red dashed line at y=0

### ✔ Why:

Residual analysis checks model assumptions:

* **Random scatter around zero** → good model
* **Patterns or curves** → model missing relationships
* **Funnel shape** → heteroscedasticity (variance issues)

### ✔ What it shows:

Scatter plot of prediction errors. Ideally:
* Points randomly scattered
* Centered around zero
* No systematic patterns

---

# ✅ **STEP 15 — Histogram of SalePrice**

### ✔ What we did:

Plotted distribution of **SalePrice** (target variable).

### ✔ Why:

Understand target variable distribution:

* Is it normally distributed?
* Are there outliers?
* Is it skewed?

This affects model performance and assumptions.

### ✔ What it shows:

Histogram with 40 bins showing price distribution. Typically shows right-skewed distribution (few very expensive houses).

---

# 🎉 **Summary (Very Helpful for Your Report)**

| Step | What We Did | Why We Did It | What It Shows |
|------|-------------|---------------|---------------|
| 1 | Imported libraries | ML tools & visualization | Setup complete |
| 2 | Loaded data | Train and test CSVs | House price dataset |
| 3 | Selected features | Focus on relevant columns | GrLivArea, BedroomAbvGr, SalePrice |
| 4 | Train/validation split | Evaluate on unseen data | 80-20 split |
| 5 | Trained model | Learn price relationships | Coefficients & intercept |
| 6 | Validated model | Measure performance | MSE, RMSE, R² scores |
| 7 | Final model training | Use all training data | Refined model |
| 8 | Predicted test data | Real-world application | Price predictions |
| 9 | Saved predictions | Output results | predictions.csv |
| 10 | Regression line plot | Visualize fit | Scatter + line |
| 11 | Bedrooms scatter | Explore 2nd feature | Bedroom-price relationship |
| 12 | Color-coded scatter | Multi-dimensional view | 3 variables in 2D |
| 13 | 3D scatter plot | Full visualization | All 3 dimensions |
| 14 | Residual plot | Check model assumptions | Error distribution |
| 15 | Price histogram | Understand target | Distribution shape |

---

# 🎯 **Key Outcomes**

* **Model Type:** Simple Linear Regression
* **Features Used:** 2 (GrLivArea, BedroomAbvGr)
* **Target Variable:** SalePrice
* **Dataset:** House Prices
  * Training: ~1,460 samples
  * Test: ~1,459 samples
* **Model Equation:** 
  ```
  SalePrice = intercept + β₁×GrLivArea + β₂×BedroomAbvGr
  ```
* **Evaluation Metrics:** MSE, RMSE, R²
* **Output:** predictions.csv with predicted prices for test data

---

# 📁 **Files in This Lab**

* **experiment.ipynb** - Complete regression workflow notebook
* **data/train.csv** - Training dataset (with SalePrice)
* **data/test.csv** - Test dataset (without SalePrice)
* **predictions.csv** - Model predictions output
* **requirements.txt** - Python dependencies
* **Plots/** - Saved visualization images
* **experiment.html** - HTML export of notebook

---

# 💡 **Key Insights**

1. **Living Area Impact:** Larger homes generally sell for higher prices (positive coefficient)
2. **Bedroom Paradox:** More bedrooms can decrease price when holding area constant (suggests smaller room sizes)
3. **Model Performance:** R² score indicates how much variance the model explains
4. **Residual Analysis:** Random scatter confirms linear assumptions are reasonable
5. **Price Distribution:** Typically right-skewed with high-value outliers

---

# 🚀 **Difference from Multiple Linear Regression (Lab 3)**

| Aspect | Lab 2 (Simple) | Lab 3 (Multiple) |
|--------|----------------|------------------|
| Features | 2 features | 11 features |
| Model Complexity | Simple relationship | Complex interactions |
| Use Case | Basic prediction | Comprehensive analysis |
| Dataset | House prices | Exam scores |

---
