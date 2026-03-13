import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sales_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)
print("Duplicates Removed")

# ---------- OUTLIER DETECTION ----------

numeric_df = df.select_dtypes(include='number')

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)

IQR = Q3 - Q1

df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Outliers Removed")

# ---------- VISUALIZATION ----------

plt.figure()
sns.boxplot(data=numeric_df)
plt.title("Outlier Detection")
plt.savefig("boxplot.png")
plt.show()

plt.figure()
df.hist()
plt.suptitle("Data Distribution")
plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ---------- SAVE CLEAN DATA ----------

df.to_csv("cleaned_dataset.csv", index=False)

print("Cleaned dataset saved successfully")