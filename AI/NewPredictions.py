import http.client, urllib.parse
import json
import pandas as pd
import joblib
import nltk
from Model import cleaning_stopwords, cleaning_punctuations, cleaning_repeating_char, cleaning_URLs, cleaning_numbers, word_tokenize, stemming_on_text, lemmatizer_on_text

conn = http.client.HTTPConnection('api.mediastack.com')

params = urllib.parse.urlencode({
    'access_key': '92326870a786707523889fb694ad68e2', #You will need your own access key
    'categories': '-general, business',
    'countries': 'us',
    'languages': 'en',
    'keywords': 'Bitcoin',
    'sort': 'published_desc',
    'limit': 100,
    })

conn.request('GET', '/v1/news?{}'.format(params))

res = conn.getresponse()
data = res.read().decode('utf-8')

json_data = json.loads(data)


articles = []
for article in json_data.get('data', []):
    author = article.get('author', '')
    title = article.get('title', '')
    description = article.get('description', '')
    url = article.get('url', '')
    source = article.get('source', '')
    image = article.get('image', '')
    category = article.get('category', '')
    language = article.get('language', '')
    country = article.get('country', '')
    published_at = article.get('published_at', '')

    articles.append({
        'Author': author,
        'Title': title,
        'Description': description,
        'URL': url,
        'Source': source,
        'Image': image,
        'Category': category,
        'Language': language,
        'Country': country,
        'Published_at': published_at
    })


df = pd.DataFrame(articles)

df.to_csv('news_data.csv', index=False)

#comment the above section out if you just want to use the pre-existing news_data.csv

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

new_data = pd.read_csv('news_data.csv')

def preprocess_text(text):
    text = text.lower()
    text = cleaning_stopwords(text)
    text = cleaning_punctuations(text)
    text = cleaning_repeating_char(text)
    text = cleaning_URLs(text)
    text = cleaning_numbers(text)
    text = word_tokenize(text)
    text = stemming_on_text(text)
    text = lemmatizer_on_text(text)
    return " ".join(text)


new_data['Description'] = new_data['Description'].apply(lambda x: preprocess_text(x))

# Load models and vectorizer
bnb_model = joblib.load('BNBmodel.pkl')
svc_model = joblib.load('SVCmodel.pkl')
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Vectorize the new data
new_data_vectorized = vectorizer.transform(new_data['Description'])

# Make predictions
bnb_predictions = bnb_model.predict(new_data_vectorized)
svc_predictions = svc_model.predict(new_data_vectorized)

# Add predictions to the new_data DataFrame
new_data['bnb_predictions'] = bnb_predictions
new_data['svc_predictions'] = svc_predictions

# Save predictions to a new CSV file
new_data.to_csv('predictions.csv', index=False)