import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Setting pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
# Directory containing CSV files
directory = 'E:\CSU Work\Thesis and RA\datasets\datasets'

# Function to process a single CSV file
def process_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, encoding_errors='ignore', skiprows=2, skipfooter=1,engine='python')
    print("Size of original Dataset of file :",filename,"is", df.shape)

    # Select columns with useful data
    df = df[['Year', '18-44 - Percentage', '45-64 - Percentage', '65-74 - Percentage', '75+ - Percentage']]
    print("\nColumn data :\n", df.head())

    # Separate data before and after 2018
    #Considering below 2019 as before covid and after 2018 as After covid
    before_2018 = df[df['Year'] <= 2018]
    after_2018 = df[df['Year'] > 2018]
    before_2018.set_index('Year', inplace=True)
    after_2018.set_index('Year', inplace=True)

    # Calculate percentage change for each age range
    def calculate_percentage_change(data):
        percent_change = {}
        age_ranges = ['18-44', '45-64', '65-74', '75+']
        for age_range in age_ranges:
            percent_change[age_range] = (data[f'{age_range} - Percentage'].iloc[-1] -
                                            data[f'{age_range} - Percentage'].iloc[0]) / \
                                           data[f'{age_range} - Percentage'].iloc[0] * 100
        return percent_change

    percent_change_before_covid = calculate_percentage_change(before_2018)
    percentage_change_after = calculate_percentage_change(after_2018)

    # Visualize percentage change for each age range
    plt.figure(figsize=(10, 6))
    plt.bar(percent_change_before_covid.keys(), percent_change_before_covid.values(), color='blue', alpha=0.7,
            label='Before COVID-19')
    plt.bar(percentage_change_after.keys(), percentage_change_after.values(), color='red', alpha=0.7,
            label='After COVID-19')
    plt.title('Percentage Change in Diabetic Cases Before and After COVID-19 by Age Range')
    plt.xlabel('Age Range')
    plt.ylabel('Percentage Change')
    plt.legend()
    plt.savefig(filename+"_percentage.png")
    plt.close()

    # Calculate correlation using pearson correlation technique
    correlation_before = before_2018.corr()
    correlation_after = after_2018.corr()

    # Visualize correlation matrices
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(correlation_before, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    plt.title('Correlation Matrix Before COVID-19')
    plt.subplot(1, 2, 2)
    sns.heatmap(correlation_after, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    plt.title('Correlation Matrix After COVID-19')
    plt.tight_layout()
    plt.savefig(filename + "_CorrelationMatrix.png")
    plt.close()

    # Calculate correlation difference for better understaning
    correlation_difference = correlation_after - correlation_before
    print("Correlation before covid is ",correlation_before)
    print("Correlation after covid is ", correlation_after)
    print("Correlation difference is ", correlation_difference)
    # Visualize correlation difference matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_difference, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Correlation Matrix Difference: After - Before COVID-19')
    plt.tight_layout()
    plt.savefig(filename + "_CorrelationDifference.png")
    plt.close()

# Process each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        process_csv(file_path)
        print("Processed ",filename,"Successfully")

