import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed dataset from the CSV file
df = pd.read_csv('/Users/saiganeshreddykodekandla/Documents/Projects/7th-Semester_project/Energy-Consumption-Prediction/results/optimization.csv')

# Get an overview of the dataset
print(df.head())  # show the first few rows
print(df.info())  # show data types and summary statistics
print(df.describe())  # show summary statistics

# Visualize the dataset

# 1. Histograms for numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    plt.hist(df[col], bins=50)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 2. Bar charts for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df)
    plt.title(f'Bar Chart of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# 3. Scatter plots for relationships between numerical columns
numerical_cols_pairs = [(col1, col2) for col1 in numerical_cols for col2 in numerical_cols if col1 != col2]
for col1, col2 in numerical_cols_pairs:
    plt.scatter(df[col1], df[col2])
    plt.title(f'Scatter Plot of {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

# 4. Heatmap for correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()