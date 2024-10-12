# Sentiment Analysis of IMDB Dataset

This project performs sentiment analysis on the IMDB movie reviews dataset. It uses natural language processing techniques to clean and lemmatize the reviews and trains machine learning models to classify the reviews as positive or negative.

## Features:
- Data Cleaning: The dataset is cleaned by removing duplicates and processing the text (lowercasing, lemmatization, removing stopwords, URLs, and special characters).
- Data Visualization: The most frequent positive and negative words are visualized using bar charts.
- Modeling: Logistic Regression and LinearSVC are trained using TfidfVectorizer to predict sentiment.
- Evaluation: Confusion matrices and accuracy scores are plotted for each model.


## Installation
Inorder to run this implementation, clone the repository by running:

-       git clone https://github.com/leonard-sanya/Future_ML_2.git
  
## Requirements

Install the necessary dependencies using the following command:

-     pip install -r requirements.txt
Download required NLTK data and model:
-       import nltk
-       nltk.download('punkt')
-       nltk.download('punkt_tab')
-       python -m spacy download en_core_web_sm (Run in the terminal)


And finally run:
-       python main.py

## Results
<img src="https://github.com/leonard-sanya/Future_ML_2/blob/main/output_images/confusion_matrix.png" width="720" height="480"/>

## License

This project is licensed under the [MIT License](LICENSE.md). Please read the License file for more information.

Feel free to explore each lab folder for detailed implementations, code examples, and any additional resources provided. Reach out to me via [email](lsanya@aimsammi.org) in case of any question or sharing of ideas and opportunities
