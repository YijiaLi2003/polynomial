from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Fetch dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize vectorizer
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
X_tfidf = vectorizer.fit_transform(documents)

# Apply SVD to reduce dimensionality
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Vectorize the query
    query_tfidf = vectorizer.transform([query])
    # Reduce dimensionality of the query
    query_reduced = svd.transform(query_tfidf)
    # Compute cosine similarities
    similarities = cosine_similarity(query_reduced, X_reduced)[0]
    # Get top 5 documents
    top_indices = similarities.argsort()[::-1][:5]
    top_documents = [documents[i] for i in top_indices]
    top_similarities = [float(similarities[i]) for i in top_indices]
    top_indices = top_indices.tolist()
    return top_documents, top_similarities, top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents_result, similarities, indices = search_engine(query)
    return jsonify({'documents': documents_result, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
