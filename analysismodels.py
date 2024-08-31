import pandas as pd
from sqlalchemy import create_engine
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from bertopic import BERTopic

from huggingface_hub import login

# Login to Hugging Face
login(token='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')

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load data from SQLite database
DATABASE_URL = "sqlite:///reddit_data.db"
engine = create_engine(DATABASE_URL)
df = pd.read_sql("SELECT * FROM reddit_posts", engine)

# Preprocess text for LDA
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

df['processed_title'] = df['title'].apply(preprocess)
dictionary = corpora.Dictionary(df['processed_title'])
corpus = [dictionary.doc2bow(text) for text in df['processed_title']]

# Load and apply sentiment analysis model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def get_sentiment(title):
    result = sentiment_analyzer(title)
    return result[0]['label'], result[0]['score']

df['sentiment'], df['sentiment_score'] = zip(*df['title'].apply(get_sentiment))

# Apply LDA
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
topics = lda_model.print_topics(num_words=4)
print("LDA Topics:")
for topic in topics:
    print(topic)

# Apply BERTopic
documents = df['title'].tolist()
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(documents)
topic_info = topic_model.get_topic_info()
print("BERTopic Information:")
print(topic_info)
