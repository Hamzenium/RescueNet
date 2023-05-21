import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_excel('/Users/muhammadhamzasohail/Desktop/Dataset-Huggingface.xlsx')
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

train_dataset = train_data.drop(columns=["Training set"])
train_labels = train_data.drop(columns=["Labels"])
test_dataset = train_data.drop(columns=["Labels"])
test_labels = train_data.drop(columns=["Training set"])

