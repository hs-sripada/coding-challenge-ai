# hello world
# print("Hello, World!")

# Step 1: Set Up Your Environment
# pip install pandas scikit-learn transformers datasets torch
# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments as HFTrainingArguments
from datasets import Dataset
import torch

# Step 2: Load & Prepare the Dataset

df = pd.read_csv('coding-challenge-ai/train.csv')  # Contains columns like 'comment_text' and labels like 'toxic'
# Limit training to first 2000 data points
df = df.head(2000)

# Create a binary label: 1 if toxic, 0 if safe
df['label'] = df['toxic'].apply(lambda x: 1 if x > 0.5 else 0)
df = df[['comment_text', 'label']].rename(columns={'comment_text': 'text'})

# Step 3: Tokenize the Text
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize)

# Step 4: Split Dataset
train_test = dataset.train_test_split(test_size=0.2)
train_ds = train_test['train']
test_ds = train_test['test']

# Step 5: Load Pre-trained Model
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)

# Step 6: Set Up Training Arguments
training_args = HFTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

# Step 7: Train the model - skip this step for now
print("Starting training...")
trainer.train()

# Step 8: Evaluate the model
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Skip training and evaluation for now
# print("Skipping training and evaluation, loading pre-trained model from checkpoint...")

# Load the model from the latest checkpoint
model = AutoModelForSequenceClassification.from_pretrained("results/checkpoint-23937")

# Step 9: Make predictions on a sample
print("Making predictions on a sample...")
sample_texts = [
    "You are a complete idiot and I hope you fail.",
    "I hate you and everything you stand for.",
    "People like you should not be allowed to speak in public.",
    "Your opinion is worthless and so are you.",
    "This group of people is ruining our country and should be removed."
]

# Tokenize the samples
inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_labels = torch.argmax(predictions, dim=-1)

# Print results
print("\nSample Predictions:")
for text, pred, probs in zip(sample_texts, predicted_labels, predictions):
    label = "unsafe" if pred == 1 else "safe"
    confidence = probs[pred].item() * 100
    print(f"Text: '{text}'")
    print(f"Prediction: {label} (Confidence: {confidence:.2f}%)\n")

print("Model training and evaluation complete!")

