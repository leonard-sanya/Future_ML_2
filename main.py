import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import torch
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab') 
import plotly.express as px
import emoji
import contractions
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

data = pd.read_csv("./data/IMDB Dataset.csv")
data.head()
data.describe()

print(' ')
print(f'====== Number of Duplicates =======')
print(len(data[data.duplicated()]))

# Remove duplicates
data = data[~data.duplicated()]


data.sentiment.value_counts().plot(kind='bar')
nlp = spacy.load('en_core_web_sm')
stopwords_list = stopwords.words('english')

def clean_lemmatize(text):
    text = emoji.demojize(text)
    text = contractions.fix(text)
    text = re.sub(r'https?://(www\.)?[\w-]+(\.[a-zA-Z]+)+|<.*?>|[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords_list and len(word) > 1]
    words = nlp(" ".join(tokens))
    text = [word.lemma_ for word in words]
    return ' '.join(text)

# Clean text
data['review'] = data['review'].apply(clean_lemmatize)
positive_words = data[data['sentiment'] == 'positive']
negative_words = data[data['sentiment'] == 'negative']


positive_words_count = pd.DataFrame(Counter(word_tokenize(' '.join(positive_words['review']))).most_common(), columns=['Words', 'Count'])
negative_words_count = pd.DataFrame(Counter(word_tokenize(' '.join(negative_words['review']))).most_common(), columns=['Words', 'Count'])

# Plot top 10 most frequent words in positive sentiment
px.bar(data_frame=positive_words_count[:10], x='Words', y='Count', title='Positive Word Frequency', color='Words')
px.bar(data_frame=negative_words_count[:10], x='Words', y='Count', title='Negative Word Frequency', color='Words')
X, y = data.drop('sentiment', axis=1), data['sentiment']

# Encode targets
encoder = LabelEncoder()
y_changed = encoder.fit_transform(y)

# Define models
models = {
    'lg': LogisticRegression(),
    'lsvc': LinearSVC(),
}


cv = StratifiedShuffleSplit(n_splits=5)
confusion_matrix_scores = {}
accuracy_scores = {}

print(' ')
print('======= Training ========')
for model_name, model in models.items():
    accuracy_list = []
    for idx, (train_index, test_index) in enumerate(cv.split(X, y_changed)):
        X_train, X_test = X.iloc[train_index].values.ravel(), X.iloc[test_index].values.ravel()
        y_train, y_test = y_changed[train_index], y_changed[test_index]
        
        print(f'{idx + 1} Split')
        
        pipe = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', model)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        
        accuracy = accuracy_score(y_pred, y_test)
        cm = confusion_matrix(y_pred, y_test)
        
        print(f'Model {model_name}\nAccuracy score: {accuracy}')
        
        if model_name not in confusion_matrix_scores.keys():
            confusion_matrix_scores[model_name] = cm
        
        accuracy_list.append(accuracy)
    
    accuracy_scores[model_name] = np.mean(accuracy_list)

def plot_confusion_matrix(cm_list, model_names_list):
    plt.figure(figsize=(20, 5))
    for i in range(len(cm_list)):
        plt.subplot(1, len(cm_list), i+1)
        sns.heatmap(cm_list[i], annot=True, fmt='.0f')
        plt.title(model_names_list[i])
        plt.tight_layout()
    plt.savefig(f'{output_folder}/confusion_matrix.png')
    plt.show()


plot_confusion_matrix(list(confusion_matrix_scores.values()), list(confusion_matrix_scores.keys()))


def plot_accuracy(models, accuracy, title):
    plt.figure(figsize=(20, 5))
    plt.bar(x=models, height=accuracy)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.savefig(f'{output_folder}/accuracy_scores.png')
    plt.show()


plot_accuracy(list(accuracy_scores.keys()), list(accuracy_scores.values()), 'Model Accuracies')
