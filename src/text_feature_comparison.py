import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from data_preprocessing import read_and_prepare_data

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_evaluation_results(file_path, model_name, y_true, y_pred):
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, 'w') as file:
        file.write(f"{model_name} Model:\n")
        accuracy = accuracy_score(y_true, y_pred)
        classification_report_text = classification_report(y_true, y_pred)
        file.write(f"Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(classification_report_text)
    print(f"{model_name} Model evaluation results saved to '{file_path}'")
def train_tfidf_model(X_train_tfidf, y_train, X_test_tfidf, y_test, save_evaluations = True, eval_file_path="../tests_tmp/evaluation_results_tfidf.txt"):
    model_tfidf = LogisticRegression()
    model_tfidf.fit(X_train_tfidf, y_train)
    
    y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
    if save_evaluations:
        write_evaluation_results(eval_file_path, 'TF-IDF', y_test, y_pred_tfidf)

def train_transformers_model(X_train_transformers_std, y_train, X_test_transformers_std, y_test, save_evaluations = True, eval_file_path="../tests_tmp/evaluation_results_transformers.txt"):
    model_transformers = LogisticRegression()
    model_transformers.fit(X_train_transformers_std, y_train)
    
    y_pred_transformers = model_transformers.predict(X_test_transformers_std)
    
    if save_evaluations:
        write_evaluation_results(eval_file_path, 'Transformers', y_test, y_pred_transformers)

def train_word2vec_model(X_train_word2vec_std, y_train, X_test_word2vec_std, y_test, save_evaluations = True, eval_file_path="../tests_tmp/evaluation_results_word2vec.txt"):
    model_word2vec = LogisticRegression()
    model_word2vec.fit(X_train_word2vec_std, y_train)
    
    y_pred_word2vec = model_word2vec.predict(X_test_word2vec_std)
    
    if save_evaluations:
        write_evaluation_results(eval_file_path, 'Word2Vec', y_test, y_pred_word2vec)

def main():
    filename = '../data/Womens Clothing E-Commerce Reviews.csv'

    df = read_and_prepare_data(filename)
    text_column = "processed_text"

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df.head(100), test_size=0.2, random_state=42)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df[text_column].fillna(''))
    X_test_tfidf = tfidf_vectorizer.transform(test_df[text_column].fillna(''))
    
    print("TF-IDF DONE")

    # Transformers (DistilBERT embeddings)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    embed_pipeline = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    
    print("Transformers initialized")

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

    # Train and evaluate models
    #evaluation_results_directory = '/test_results/'
    train_tfidf_model(X_train_tfidf, train_df['recommended_ind'], X_test_tfidf, test_df['recommended_ind'])#, evaluation_results_directory)
    train_transformers_model(X_train_transformers_std, train_df['recommended_ind'], X_test_transformers_std, test_df['recommended_ind'])#, evaluation_results_directory)

    # Word2Vec
    tokenized_sentences_train = train_df['review_text'].apply(lambda x: str(x).split())
    word2vec_model = Word2Vec(sentences=tokenized_sentences_train, vector_size=100, window=5, min_count=1, workers=4)
    
    X_train_word2vec_std = train_df['review_text'].apply(lambda x: [word2vec_model.wv[word] for word in str(x).split() if word in word2vec_model.wv]).apply(lambda x: sum(x) / len(x) if len(x) > 0 else np.zeros(100))
    X_train_word2vec_std = np.vstack(X_train_word2vec_std)
    
    X_test_word2vec_std = test_df['review_text'].apply(lambda x: [word2vec_model.wv[word] for word in str(x).split() if word in word2vec_model.wv]).apply(lambda x: sum(x) / len(x) if len(x) > 0 else np.zeros(100))
    X_test_word2vec_std = np.vstack(X_test_word2vec_std)

    scaler = StandardScaler()
    X_train_word2vec_std = scaler.fit_transform(X_train_word2vec_std)
    X_test_word2vec_std = scaler.transform(X_test_word2vec_std)

    # Train and evaluate Word2Vec model
    train_word2vec_model(X_train_word2vec_std, train_df['recommended_ind'], X_test_word2vec_std, test_df['recommended_ind'])#, evaluation_results_directory)

if __name__ == "__main__":
    main()
