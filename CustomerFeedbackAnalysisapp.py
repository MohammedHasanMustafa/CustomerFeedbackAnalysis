from flask import Flask, request, render_template, redirect, url_for
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_path = r'C:\Users\CSE_BAY3\CustomerFeedbackAnalysis\fine-tuned-model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    category = request.form.get('category')

    # Simple validation
    if not name or not email or not message or not category:
        return render_template('index.html', error='All fields are required.')

    # Tokenize and predict
    inputs = tokenizer(message, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    sentiment = 'positive' if prediction == 1 else 'negative'
    confidence = torch.softmax(logits, dim=1).max().item()

    # Render results
    return render_template('index.html', 
                           name=name, 
                           email=email, 
                           message=message, 
                           category=category,
                           sentiment=sentiment,
                           confidence=confidence,
                           success='Form submitted successfully!')

if __name__ == '__main__':
    app.run(debug=True)
