import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Extract and scale the age feature
df['age'] = df['age'] * 100  # Scaling the age feature

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=20, edgecolor='black', color='skyblue')
plt.xlabel('Age (scaled)')
plt.ylabel('Frequency')
plt.title('Age Distribution in the Diabetes Dataset')
plt.show()