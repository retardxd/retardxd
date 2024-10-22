# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection
# Load the dataset into a DataFrame
url = "diabetes.csv"  # Replace with the actual URL or file path
data = pd.read_csv(url)

# Step 2: Data Cleaning
# Display the first few rows of the dataset
print(data.head())

# Check for missing values in the dataset
print(data.isnull().sum())

# Handle missing values by filling with the mean of the column
data.fillna(data.mean(), inplace=True)

# Remove duplicate rows, if any
data.drop_duplicates(inplace=True)

# Step 3: Data Integration
# Example of integrating another dataset if available (commented out as an example)
# additional_data = pd.read_csv("https://example.com/additional_data.csv")
# data = pd.merge(data, additional_data, on="common_column")

# Step 4: Data Transformation
# Normalize the data (scaling features)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['Outcome']))  # Assuming 'Outcome' is the target column

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=data.columns[:-1])
scaled_df['Outcome'] = data['Outcome']

# Display the first few rows of the transformed dataset
print(scaled_df.head())

# Step 5: Exploratory Data Analysis (EDA)
# Plotting the distribution of 'Age' and 'BMI'
plt.figure(figsize=(12, 6))
sns.histplot(data=scaled_df, x='Age', kde=True, bins=30, color='blue', label='Age')
sns.histplot(data=scaled_df, x='BMI', kde=True, bins=30, color='red', label='BMI')
plt.legend()
plt.title('Distribution of Age and BMI')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Correlation analysis
correlation_matrix = scaled_df.corr()
print(correlation_matrix)

# Visualizing the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
