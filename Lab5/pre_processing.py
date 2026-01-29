import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load the dataset
# -----------------------------
FILE_PATH = "./iris_data.csv"   # change if your file name is different
df = pd.read_csv(FILE_PATH)

print("Initial dataset shape:", df.shape)
print("\nInitial columns:")
print(df.columns)

# -----------------------------
# 2. Rename columns (clean format)
# -----------------------------
df.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "label"
]

# -----------------------------
# 3. Check data types
# -----------------------------
print("\nData types:")
print(df.dtypes)

# -----------------------------
# 4. Check for missing values
# -----------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()
print("\nShape after removing missing values:", df.shape)

# -----------------------------
# 5. Check for duplicate rows
# -----------------------------
duplicate_count = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_count)

# Remove duplicates
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# -----------------------------
# 6. Separate features and labels
# -----------------------------
X = df.drop(columns=["label"])
y = df["label"]

# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 8. Combine back and save files
# -----------------------------
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nFiles saved successfully:")
print("- train.csv")
print("- test.csv")

print("\nFinal Train shape:", train_df.shape)
print("Final Test shape:", test_df.shape)
