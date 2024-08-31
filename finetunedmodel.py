import sqlite3
import pandas as pd
from datasets import Dataset, load_metric
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# Step 1: Extract Data from SQLite Database
def extract_data_from_db(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT title, score FROM reddit_posts"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    df['title'] = df['title'].str.lower().str.strip()
    df['label'] = df['score'].apply(lambda x: 1 if x > 0 else 0)  # Example: positive (1) or negative (0)
    return df

# Step 3: Prepare Data for Fine-Tuning
def prepare_data(df):
    dataset = Dataset.from_pandas(df[['title', 'label']])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['title'], padding='max_length', truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

# Step 4: Fine-Tune the Model
def fine_tune_model(dataset):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    metric = load_metric('accuracy', trust_remote_code=True)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    return trainer

# Step 5: Evaluate and Save the Model
def evaluate_and_save_model(trainer):
    results = trainer.evaluate()
    print(results)
    
    model_path = 'fine-tuned-model'
    trainer.save_model(model_path)
    trainer.save_state()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.save_pretrained(model_path)

# Step 6: Use the Fine-Tuned Model
def predict_sentiment(text, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    return predicted_class

# Main script
if __name__ == "__main__":
    db_path = 'reddit_data.db'
    df = extract_data_from_db(db_path)
    df = preprocess_data(df)
    dataset = prepare_data(df)
    
    # Split the dataset into train and test
    dataset = dataset.train_test_split(test_size=0.2)
    
    trainer = fine_tune_model(dataset)
    evaluate_and_save_model(trainer)
    
    # Example prediction
    text = "I love learning new things about Python!"
    model_path = 'fine-tuned-model'
    sentiment = predict_sentiment(text, model_path)
    print(f"Predicted sentiment: {sentiment}")

