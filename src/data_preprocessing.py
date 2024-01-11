import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def read_data(filename):

    df = pd.read_csv(filename)
    df.drop(columns=['Unnamed: 0'],inplace=True)

    return df

def make_pythonic_column_names(df):
    """
    Convert all column names in the DataFrame to a Pythonic format.
    """
    new_column_names = []

    for column_name in df.columns:
        # Convert spaces to underscores and make lowercase
        pythonic_name = column_name.lower().replace(' ', '_')
        new_column_names.append(pythonic_name)

    df.columns = new_column_names
    return df

def preprocessing(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Apply stemming (you can also consider lemmatization)
    ps = PorterStemmer()
    text = " ".join([ps.stem(word) for word in text.split()])
    return text


def read_and_prepare_data(filename):

    df = read_data(filename=filename)
    df = make_pythonic_column_names(df)

    df = df.dropna(subset=['title', 'review_text'])
    df = df.drop_duplicates()

    df["processed_text"] = df["review_text"].apply(preprocessing)

    return df

