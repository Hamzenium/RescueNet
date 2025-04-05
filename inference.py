from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np

model_checkpoint = "./results/checkpoint-250"

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

model = RobertaForSequenceClassification.from_pretrained(model_checkpoint)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['E'])  # Or include all classes: np.array(['E', 'N', 'X'])


text = "I'm happy"

result = classifier(text)
predicted_index = int(result[0]['label'].split('_')[-1])
predicted_label = label_encoder.inverse_transform([predicted_index])[0]

print(f"🧠 Prediction: {predicted_label}")
