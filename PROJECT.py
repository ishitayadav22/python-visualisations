import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Load data
df = pd.read_csv("C:/Users/yadav/Downloads/1000 Sales Records.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Objective: Basic exploration using head, tail, info, describe
print("Dataset Preview:")
print(df.head())

print("\nLast Rows:")
print(df.tail())

print("\nDataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nUnique Values in Key Columns:")
print("Item Types:", df['Item Type'].nunique())
print("Countries:", df['Country'].nunique())
print("Sales Channels:", df['Sales Channel'].unique())
print("Order Priorities:", df['Order Priority'].unique())

# Objective: Compare the performance of online vs. offline sales channels
online_profit = df[df['Sales Channel'] == 'Online']['Total Profit']
offline_profit = df[df['Sales Channel'] == 'Offline']['Total Profit']

print("\nDescriptive Statistics for Sales Channels:")
print("Online Profit Mean:", online_profit.mean())
print("Offline Profit Mean:", offline_profit.mean())
print("Online Profit Std Dev:", online_profit.std())
print("Offline Profit Std Dev:", offline_profit.std())

sample_size = min(len(online_profit), len(offline_profit), 500)

shapiro_online = stats.shapiro(online_profit.sample(sample_size, random_state=1))
shapiro_offline = stats.shapiro(offline_profit.sample(sample_size, random_state=1))
print("\nShapiro-Wilk Test for Normality:")
print(f"Online Profit: W={shapiro_online.statistic:.4f}, p={shapiro_online.pvalue:.4f}")
print(f"Offline Profit: W={shapiro_offline.statistic:.4f}, p={shapiro_offline.pvalue:.4f}")

levene_test = stats.levene(online_profit, offline_profit)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic={levene_test.statistic:.4f}, p={levene_test.pvalue:.4f}")

t_stat, p_value = stats.ttest_ind(online_profit, offline_profit, equal_var=False)
print("\nIndependent T-Test Result:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject the null hypothesis. Significant difference exists.")
else:
    print("Conclusion: Fail to reject the null hypothesis. No significant difference.")

# Objective: Visualize the difference in profit distribution between channels
plt.figure(figsize=(8, 5))
sns.boxplot(x='Sales Channel', y='Total Profit', data=df, hue='Sales Channel', palette='coolwarm', legend=False)
plt.title("Distribution of Total Profit by Sales Channel")
plt.xlabel("Sales Channel")
plt.ylabel("Total Profit")
plt.tight_layout()
plt.show()

# Objective: Determine which item type generates the highest total profit
item_profit = df.groupby('Item Type')['Total Profit'].sum().sort_values(ascending=False)
top_items = item_profit.head(5)
plt.figure(figsize=(6, 6))
plt.pie(top_items, labels=top_items.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("muted"))
plt.title("Top 5 Item Types by Total Profit")
plt.ylabel("Item Types")
plt.tight_layout()
plt.show()

# Objective: Identify the most profitable and least profitable countries
country_profit = df.groupby('Country')['Total Profit'].sum().sort_values(ascending=False)
top_countries = country_profit.head(10).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=top_countries, x='Total Profit', y='Country', hue='Country', legend=False, palette='Spectral')
plt.title('Top 10 Most Profitable Countries')
plt.xlabel("Total Profit")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# Objective: Analyze total revenue, cost, and profit across different regions
region_summary = df.groupby('Region')[['Total Revenue', 'Total Cost', 'Total Profit']].sum()
plt.figure(figsize=(8, 6))
sns.heatmap(region_summary, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Revenue, Cost, and Profit by Region")
plt.xlabel("Region")
plt.ylabel("Amount")
plt.tight_layout()
plt.show()

# Objective: Study the impact of order priority on total revenue and profit
priority_summary = df.groupby('Order Priority')[['Total Revenue', 'Total Profit']].sum()
plt.figure(figsize=(8, 5))
sns.lineplot(data=priority_summary, markers=True, palette='Set2')
plt.title("Revenue and Profit by Order Priority")
plt.xlabel("Order Priority")
plt.ylabel("Amount")
plt.tight_layout()
plt.show()

# Objective: Scatter plot for Total Revenue vs Total Profit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Total Revenue', y='Total Profit', hue='Sales Channel', palette='coolwarm', alpha=0.7)
plt.title("Scatter Plot: Total Revenue vs Total Profit")
plt.xlabel("Total Revenue")
plt.ylabel("Total Profit")
plt.tight_layout()
plt.show()
