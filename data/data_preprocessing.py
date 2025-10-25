import sqlite3
import pandas as pd


df = pd.read_csv(r"C:\Users\ziovi\Dropbox\PC\Downloads\creditcard.csv (1)\creditcard.csv")


df = df.drop(columns=['Time'])


print(df.isnull().sum())


print(df['Class'].value_counts())


conn = sqlite3.connect('data/creditcard_fraud.db')


df.to_sql('transactions', conn, if_exists='replace', index=False)

print("Data successfully saved to SQLite database!")


conn.close()
