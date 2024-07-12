import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target  # Add the target variable to the DataFrame

# Display the first few rows of the dataset
print(df.head())

# Display summary statistics
print(df.describe())

# Display data types and non-null counts
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Plot histograms for key features and the target variable
key_features = ['age', 'bmi', 'bp']
df[key_features + ['target']].hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Key Features and Target Variable')
plt.show()

# Scatter plots of key features vs target variable
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Scatter plots of Key Features vs Target Variable')

for i, col in enumerate(key_features):
    sns.scatterplot(data=df, x=col, y='target', ax=axs[i], color='skyblue')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Compute the correlation matrix for key features and target
key_features.append('target')
corr = df[key_features].corr()

# Generate a heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Key Features and Target')
plt.show()