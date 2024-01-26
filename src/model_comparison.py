from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import write_evaluation_results
from data_preprocessing import read_and_prepare_data
from prepare_embeddings import prepare_tfidf_embeddings, prepare_transformers_embeddings, prepare_word2vec_embeddings
from loguru import logger
import numpy as np

filename = '../data/Womens Clothing E-Commerce Reviews.csv'
preprocess_text = True

if preprocess_text==True:
    text_column = "processed_text"
    filename_specification = "preprocessed"
else:
    text_column = "review_text"
    filename_specification = "unpreprocessed"

df = read_and_prepare_data(filename, preprocess=preprocess_text)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

logger.info("Preparing the TF-IDF vectors.")
X_train_tfidf, X_test_tfidf = prepare_tfidf_embeddings(train_df, test_df, text_column)

#logger.info("Preparing the Transformers vectors.")
#X_train_transformers_std, X_test_transformers_std = prepare_transformers_embeddings(train_df, test_df, text_column)

logger.info("Reading the Transformers vectors.")
X_train_transformers = np.load('../data/X_train_transformers.npy')
X_test_transformers = np.load('../data/X_test_transformers.npy')

logger.info("Preparing the Word2Vec vectors.")
X_train_word2vec_std, X_test_word2vec_std = prepare_word2vec_embeddings(train_df, test_df, text_column)

logger.info("Embeddings prepared")


# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

def train_embeddings_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def analyze_models(models, X_train, X_test, y_train, y_test, output_file_path):
    # Inizializza il file all'inizio
    with open(output_file_path, 'w') as combined_file:
        combined_file.write("Combined Evaluation Results:\n\n")

        # Ciclo sui modelli
        for model_name, model in models.items():
            logger.info(f"Working with {model_name}")
            model_trained = train_embeddings_model(model, X_train, y_train)
            y_pred = model_trained.predict(X_test)
            write_evaluation_results(combined_file, model_name, y_test, y_pred)


output_file_path_tfidf = f"../tests_tmp/tfidf_evaluation_results_combined_{filename_specification}.txt"
output_file_path_word2vec = f"../tests_tmp/word2vec_evaluation_results_combined_{filename_specification}.txt"
output_file_path_transformers = f"../tests_tmp/transformers_evaluation_results_combined_{filename_specification}.txt"

# Analizza modelli per TF-IDF
analyze_models(models, X_train_tfidf, X_test_tfidf, train_df["recommended_ind"], test_df["recommended_ind"], output_file_path_tfidf)

# Analizza modelli per Word2Vec
analyze_models(models, X_train_word2vec_std, X_test_word2vec_std, train_df["recommended_ind"], test_df["recommended_ind"], output_file_path_word2vec)

# Analizza modelli per Transformers
analyze_models(models, X_train_transformers, X_test_transformers, train_df["recommended_ind"], test_df["recommended_ind"], output_file_path_transformers)