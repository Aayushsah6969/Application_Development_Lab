# 🧹 Data Preprocessing & Visualization - Exam Score Dataset

**Complete explanation** of data preprocessing steps and exploratory visualizations performed on the Exam Score Prediction dataset.

---

## 📋 **Overview**

This lab focuses on:
- Loading and inspecting raw data
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Creating visualizations to understand data patterns

---

# ✅ **STEP 1 — Import Libraries**

### ✔ What we did:

We imported all the Python libraries needed:

* **pandas** → data loading & manipulation
* **numpy** → numeric operations
* **matplotlib / seaborn** → plotting & visualization
* **sklearn.impute** → SimpleImputer for handling missing values
* **sklearn.preprocessing** → OneHotEncoder, StandardScaler
* **sklearn.compose** → ColumnTransformer for pipeline processing
* **sklearn.pipeline** → Pipeline for chaining preprocessing steps

### ✔ Why:

Each library serves a specific purpose in data preprocessing and visualization. These tools allow us to clean, transform, and analyze data efficiently.

### ✔ What it shows:

Nothing visual — it prepares the environment for data preprocessing.

---

# ✅ **STEP 2 — Load Dataset**

### ✔ What we did:

Loaded the Exam_Score_Prediction.csv file:

```python
df = pd.read_csv('Exam_Score_Prediction.csv')
```

### ✔ Why:

To bring the raw dataset into pandas for inspection, cleaning, and transformation.

### ✔ What it shows:

* First few rows of the dataset using `df.head()`
* Column names and initial data structure
* Dataset contains: student_id, age, gender, course, study_hours, class_attendance, internet_access, sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty, exam_score

---

# ✅ **STEP 3 — Inspect the Data**

### ✔ What we did:

We checked:

* Data types of each column using `df.info()`
* Missing values using `df.isna().sum()`
* Summary statistics using `df.describe(include='all')`

### ✔ Why:

Before preprocessing, we must understand:

* Which columns are numeric or categorical
* Whether any data is missing
* If values look normal (ranges, averages, etc.)
* Statistical properties of each feature

### ✔ What it shows:

* Column data types (int64, float64, object)
* Count of missing values per column
* Basic statistics: min, max, mean, median, std, unique values

**Finding:** This dataset has **no missing values** initially.

---

# ✅ **STEP 4 — Identify Column Types**

### ✔ What we did:

Separated columns into two categories:

**Numeric columns:**
* age, study_hours, class_attendance, sleep_hours, exam_score

**Categorical columns:**
* gender, course, internet_access, sleep_quality, study_method, facility_rating, exam_difficulty

### ✔ Why:

Different data types require different preprocessing techniques:
* Numeric → imputation with mean, scaling
* Categorical → imputation with mode, encoding

### ✔ What it shows:

Clear categorization of features for targeted preprocessing strategies.

---

# ✅ **STEP 5 — Handle Missing Values**

### ✔ What we did:

Applied imputation strategies (even though no missing values exist, this prepares for real-world scenarios):

* **Numeric columns** → filled missing values with **mean** using SimpleImputer
* **Categorical columns** → filled missing values with **most frequent value (mode)**

```python
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
```

### ✔ Why:

Real datasets often have missing values. Machine learning models cannot handle missing data — we must fill or remove them. This step ensures robustness.

### ✔ What it shows:

After imputation:
* The dataset remains complete with **no missing values**
* Each column is ready for further preprocessing

---

# ✅ **STEP 6 — Encode Categorical Data**

### ✔ What we did:

Converted categorical text features into **numeric one-hot encoded columns** using `pd.get_dummies()`:

* gender → gender_male, gender_other
* course → course_bca, course_b.com, course_b.sc, course_b.tech, course_diploma
* internet_access → internet_access_yes
* sleep_quality → sleep_quality_good, sleep_quality_poor
* study_method → study_method_group study, study_method_mixed, study_method_online videos, study_method_self-study
* facility_rating → facility_rating_low, facility_rating_medium
* exam_difficulty → exam_difficulty_hard, exam_difficulty_moderate

### ✔ Why:

Machine learning models cannot understand text. One-hot encoding:
* Converts categories into binary (0/1) columns
* Prevents false ordinal relationships
* Maintains category information without bias

### ✔ What it shows:

The encoded DataFrame now has **many more columns** — one binary column for each category (minus one with `drop_first=True` to avoid multicollinearity).

---

# ✅ **STEP 7 — Feature Scaling**

### ✔ What we did:

Applied **StandardScaler** to numeric features:

* age
* study_hours
* class_attendance
* sleep_hours
* exam_score

Scaling transforms values so they have:
* Mean = 0
* Standard deviation = 1

```python
scaler = StandardScaler()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
```

### ✔ Why:

Different features have different units and ranges:

* study_hours: 0–10
* age: 17–25
* class_attendance: 0–100
* exam_score: 0–100

Without scaling:
* Features with larger ranges dominate smaller ones
* Many ML algorithms (especially distance-based) perform poorly

### ✔ What it shows:

The scaled DataFrame shows standardized values:

```
-0.84, 0.21, 1.45, ...
```

All numeric features now have comparable scales.

---

# 📊 **VISUALIZATION STEPS**

# ✅ **STEP 8 — Histogram**

### ✔ What we did:

Plotted a histogram of **study_hours**.

### ✔ Why:

A histogram shows the **distribution** of a single numeric feature:

* Is the data skewed or normally distributed?
* Are most students studying few or many hours?
* Are there outliers or unusual patterns?

### ✔ What it shows:

Bars representing frequency of students in each study_hours range (15 bins).

This helps understand:
* Central tendency
* Data spread
* Distribution shape

---

# ✅ **STEP 9 — Scatter Plots**

### ✔ What we did:

Created three scatter plots to explore relationships:

1. **Study Hours vs Exam Score** (colored by gender)
2. **Class Attendance vs Exam Score** (colored by gender)
3. **Sleep Hours vs Study Hours** (colored by gender)

### ✔ Why:

Scatter plots reveal **relationships** between two numerical variables:

* Is there a positive/negative correlation?
* Do different genders show different patterns?
* Are there clusters or outliers?

### ✔ What it shows:

**Key insights:**
* Do more study hours lead to higher scores?
* Does attendance impact performance?
* Is there a trade-off between sleep and study?

Points scattered show patterns:
* Upward slope → positive relationship
* Downward slope → negative relationship
* Random scatter → weak relationship

---

# ✅ **STEP 10 — Correlation Heatmap**

### ✔ What we did:

Created two correlation heatmaps:

1. **Full encoded dataset** correlation (all features)
2. **Numeric features only** correlation (with annotation values)

### ✔ Why:

Correlation helps answer:

* Which features are most strongly related to exam_score?
* Do some features duplicate the same information (multicollinearity)?
* Which features should be prioritized for modeling?

### ✔ What it shows:

A **colored matrix** where:

* Values close to **+1** → strong positive correlation
* Values close to **-1** → strong negative correlation
* Values near **0** → no correlation

The heatmap visually highlights:
* Important predictors of exam_score
* Redundant features
* Relationships between all variables

---

# 🎉 **Summary (Very Helpful for Your Report)**

| Step | What We Did | Why We Did It | What It Shows |
|------|-------------|---------------|---------------|
| 1 | Imported libraries | Tools for preprocessing & visualization | Setup complete |
| 2 | Loaded dataset | Bring raw CSV into pandas | First rows of data |
| 3 | Inspected data | Understand structure & completeness | Data types, no missing values |
| 4 | Identified column types | Categorize numeric vs categorical | 5 numeric, 7 categorical |
| 5 | Handled missing values | Prepare for real-world scenarios | Imputation strategies applied |
| 6 | Encoded categorical data | Convert text → numeric (one-hot) | Binary encoded columns |
| 7 | Scaled features | Normalize numeric values | Mean=0, Std=1 |
| 8 | Histogram | Study distribution of study_hours | Shape of data spread |
| 9 | Scatter plots | Check relationships between variables | Patterns & correlations |
| 10 | Correlation heatmap | Find strongest relationships | Matrix of correlations |

---

# 🎯 **Key Outcomes**

* **Original Dataset:** 295 rows × 13 columns
* **After Encoding:** 295 rows × 30+ columns (with one-hot encoding)
* **Numeric Features:** age, study_hours, class_attendance, sleep_hours, exam_score
* **Categorical Features:** gender, course, internet_access, sleep_quality, study_method, facility_rating, exam_difficulty
* **Missing Values:** None detected
* **Preprocessing Complete:** Data is now ready for machine learning models

---

# 📁 **Files in This Lab**

* **preProcessing.ipynb** - Complete preprocessing workflow notebook
* **Exam_Score_Prediction.csv** - Raw dataset
* **requirements.txt** - Python dependencies
* **Plots/** - Saved visualization images
* **preProcessing.html** - HTML export of notebook
* **experiment_details** - Detailed step-by-step explanations

---

# 🚀 **Next Steps**

After preprocessing, the data is ready for:
1. Machine Learning model training (Linear Regression, Decision Trees, etc.)
2. Further feature engineering
3. Model evaluation and comparison
4. Predictive analytics

---
