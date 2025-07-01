import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Load temperature data into data frames
temp_df = pd.read_csv("E:\CSU Work\Thesis and RA\datasets_2\Annual_Surface_Temperature_Change.csv", encoding_errors='ignore')

# Load methane emissions data for India into data frames
methane_df = pd.read_csv("E:\CSU Work\Thesis and RA\datasets_2\Emissions_Country_india.csv", encoding_errors='ignore')

# Filter temperature data for Country India and drop the unnecessary columns
temp_df = temp_df[temp_df['Country'] == 'India']
temp_df = temp_df.drop(columns=['ObjectId','Country','ISO2','ISO3','Indicator','Code','Unit','Source'])

# Reset the index before transposing
temp_df.reset_index(drop=True, inplace=True)

# Transpose the DataFrame
transposed_temp_df = temp_df.transpose()

# Reset the index
transposed_temp_df.reset_index(inplace=True)

# Rename the columns
transposed_temp_df.columns = ['Year', 'Temperature']

# Convert 'Year' values to integers by Replacing F1961 with 1961
transposed_temp_df['Year'] = transposed_temp_df['Year'].str.replace('F', '').astype(int)

# Load only relevant methane emissions data for item as Rice Cultivation amd element as CH4
methane_df = methane_df[(methane_df['Item'] == 'Rice Cultivation') & (methane_df['Element'] == 'Emissions (CH4)')]
# Considering only the required Data
methane_df = methane_df[['Year', 'Value']]


# Merge temperature and methane emissions data on 'Year' column
combined_data = pd.merge(transposed_temp_df, methane_df, on='Year')
print(combined_data)

# Calculate correlation between methane emissions and temperature change
correlation = combined_data['Value'].corr(combined_data['Temperature'])
print("Correlation between Methane Emissions and Temperature Change:", correlation)

# Visualize correlation between methane Emission and temperature change
plt.scatter(combined_data['Value'], combined_data['Temperature'])
plt.title('Correlation between Methane Emissions and Temperature Change')
plt.xlabel('Methane Emissions (CH4)')
plt.ylabel('Temperature Change')
plt.savefig('CorrelationMatrix.png')

# Proceed with the Regression model as there is a correlation bw Methane Emission and Temperature change
# Split the data into training and testing sets
X = combined_data[['Value']].values
y = combined_data['Temperature'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Predict temperature change for the next 10 years
next_10_years_methane = np.arange(combined_data['Value'].max() + 1, combined_data['Value'].max() + 11).reshape(-1, 1)
next_10_years_temperature_change = regression_model.predict(next_10_years_methane)

# Print the predicted temperature change for the next 10 years
print("Predicted temperature change for the next 10 years:")
for i, year in enumerate(range(combined_data['Year'].max() + 1, combined_data['Year'].max() + 11)):
    print(f"Year {year}: {next_10_years_temperature_change[i]}")

# Calculate mean squared error
mse = mean_squared_error(y_test, regression_model.predict(X_test))
print("Mean Squared Error:", mse)

# Calculate R-squared score
r_squared = regression_model.score(X_test, y_test)
print("R-squared Score:", r_squared)


# Plot the dataset along with the predictions
plt.scatter(combined_data['Value'], combined_data['Temperature'], label='Actual Data')
plt.plot(next_10_years_methane, next_10_years_temperature_change, color='red', label='Predictions')
plt.title('Methane Emissions vs Temperature Change')
plt.xlabel('Methane Emissions (CH4)')
plt.ylabel('Temperature Change')
plt.legend()
plt.savefig("LinearRegression.png")
