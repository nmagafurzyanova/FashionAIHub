import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from utils import ensure_directory_exists
from data_preprocessing import read_and_prepare_data
from sklearn.model_selection import train_test_split
from loguru import logger

def prepare_tfidf_embeddings(train_df, test_df, text_column):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df[text_column].fillna(''))
    X_test_tfidf = tfidf_vectorizer.transform(test_df[text_column].fillna(''))
    return X_train_tfidf, X_test_tfidf

def prepare_transformers_embeddings(train_df, test_df, text_column):
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    embed_pipeline = pipeline('feature-extraction', model=model, tokenizer=tokenizer)    
   # Obtain DistilBERT embeddings
    X_train_transformers = train_df[text_column].apply(lambda x: np.vstack(embed_pipeline(x)[0]).mean(axis=0))
    X_test_transformers = test_df[text_column].apply(lambda x: np.vstack(embed_pipeline(x)[0]).mean(axis=0))
    
    print("Transformers embedding done")

    # Save the BERT embeddings
    embeddings_directory = '../tests_tmp/'
    ensure_directory_exists(embeddings_directory)
    np.save(os.path.join(embeddings_directory, 'X_train_transformers.npy'), np.vstack(X_train_transformers.apply(lambda x: x.tolist()).tolist()))
    np.save(os.path.join(embeddings_directory, 'X_test_transformers.npy'), np.vstack(X_test_transformers.apply(lambda x: x.tolist()).tolist()))

    scaler = StandardScaler()
    X_train_transformers_std = scaler.fit_transform(X_train_transformers.apply(lambda x: x.tolist()).tolist())
    X_test_transformers_std = scaler.transform(X_test_transformers.apply(lambda x: x.tolist()).tolist())

    return X_train_transformers_std, X_test_transformers_std

def prepare_word2vec_embeddings(train_df, test_df, text_column):
    tokenized_sentences_train = train_df[text_column].apply(lambda x: str(x).split())
    word2vec_model = Word2Vec(sentences=tokenized_sentences_train, vector_size=100, window=5, min_count=1, workers=4)
    
    X_train_word2vec_std = train_df[text_column].apply(lambda x: [word2vec_model.wv[word] for word in str(x).split() if word in word2vec_model.wv]).apply(lambda x: sum(x) / len(x) if len(x) > 0 else np.zeros(100))
    X_train_word2vec_std = np.vstack(X_train_word2vec_std)
    
    X_test_word2vec_std = test_df[text_column].apply(lambda x: [word2vec_model.wv[word] for word in str(x).split() if word in word2vec_model.wv]).apply(lambda x: sum(x) / len(x) if len(x) > 0 else np.zeros(100))
    X_test_word2vec_std = np.vstack(X_test_word2vec_std)

    scaler = StandardScaler()
    X_train_word2vec_std = scaler.fit_transform(X_train_word2vec_std)
    X_test_word2vec_std = scaler.transform(X_test_word2vec_std)

    return X_train_word2vec_std, X_test_word2vec_std

