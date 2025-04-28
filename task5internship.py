# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8')
# Load Titanic dataset from an online URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Show the shape of the dataset
print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
# View the first 5 rows
df.head()
# Basic information about dataset
df.info()
# Statistical description
df.describe()
# Checking for missing values
df.isnull().sum()
# Survival counts
print(df['Survived'].value_counts())

# Passenger Class counts
print(df['Pclass'].value_counts())

# Gender counts
print(df['Sex'].value_counts())
plt.figure(figsize=(8,5))
sns.histplot(df['Age'].dropna(), kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survival')
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
plt.title('Survival Count by Sex')
plt.show()
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Fare vs Age Colored by Survival')
plt.show()
plt.figure(figsize=(10,8))
# Selecting only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Then plotting
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

plt.title('Correlation Matrix')
plt.show()
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.show()
# Fill 'Age' missing values with median
df['Age'].fillna(df['Age'].median())

# Drop 'Cabin' since it's mostly missing
df.drop('Cabin', axis=1, inplace=True)

# Fill 'Embarked' with most common value
df['Embarked'].fillna(df['Embarked'].mode()[0])
