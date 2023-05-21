import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_excel('/Users/muhammadhamzasohail/Desktop/Dataset-Huggingface.xlsx')
data = data.iloc[1:]
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

print(train_data.iloc[1])