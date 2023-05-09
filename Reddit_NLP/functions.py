from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

def preprocess_text(row : str) -> str:
    '''
    Prepare text for NLP Analysis.
    INPUT:
        - row: each row of the column you will perfomr NLP analysis.
    OUTPUT:
        - cleaned_text: row ready to perform NLP analysis.
    '''
    # Tokenization
    words = word_tokenize(row)

    # Lowercasing
    words = [word.lower() for word in words]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming: removing suffixes
    # Lemmatization: converting words to their base form using a dictionary
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]

    # Delete special characters
    words = [word for word in words if word.isalpha()]

    # list to string
    cleaned_text = ' '.join(words)

    return cleaned_text


def union_dataframes(new, path, id):
    '''
    Concatenate two dataframes and erase duplicated values.
    '''
    old = pd.read_csv(path, encoding= 'unicode_escape')
    new = pd.concat([new, old])
    new['date'] = pd.to_datetime(new['date'])
    new = new.sort_values(by='date', ascending=False)
    new = new[~new[id].duplicated(keep="last")]
    return new


def sentiment_analysis(text):
    '''
    Run Sentiment Analysis on a pandas column.
    '''
    sia = SIA()
    score = sia.polarity_scores(text)
    return {
        'neg': score['neg'],
        'neu': score['neu'],
        'pos': score['pos'],
        'compound': score['compound']
    }