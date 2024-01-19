import os 
from sklearn.metrics import accuracy_score, classification_report

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_evaluation_results(file, model_name, y_true, y_pred):

    file.write(f"{model_name} Model:\n")
    accuracy = accuracy_score(y_true, y_pred)
    classification_report_text = classification_report(y_true, y_pred)
    file.write(f"Accuracy: {accuracy}\n")
    file.write("Classification Report:\n")
    file.write(classification_report_text)
    file.write("\n\n")