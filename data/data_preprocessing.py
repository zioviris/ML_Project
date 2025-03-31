import sqlite3
import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\ziovi\Dropbox\PC\Downloads\creditcard.csv (1)\creditcard.csv")

# Drop the 'Time' column
df = df.drop(columns=['Time'])

# Check for missing values
print(df.isnull().sum())

# Check class distribution
print(df['Class'].value_counts())

# Create or connect to an SQLite database
conn = sqlite3.connect('data/creditcard_fraud.db')

# Save the DataFrame to a table named 'transactions' in the SQLite database
df.to_sql('transactions', conn, if_exists='replace', index=False)

print("Data successfully saved to SQLite database!")

# Close the connection
conn.close()
