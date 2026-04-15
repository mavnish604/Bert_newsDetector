import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib
import os

# Configuration
DATA_PATH = "data/WELFake_Dataset.csv"
MODEL_PATH = "fake_news_model/checkpoint-10731"
CLASSIFIER_SAVE_PATH = "classifier.joblib"

def invert_label(val):
    if val == 1:
        return 0
    else:
        return 1

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preprocessing data...")
    # Follow the notebook's preprocessing
    df["label"] = df["label"].apply(invert_label)
    df = df.dropna()
    
    X = df["text"]
    y = df["label"]
    
    # Split as in notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    
    print(f"Loading SentenceTransformer from {MODEL_PATH}...")
    model = SentenceTransformer(MODEL_PATH)
    
    print("Generating embeddings for training set (this may take a few minutes)...")
    
    train_embeddings = model.encode(X_train.tolist(), show_progress_bar=True)
    
    print("Training GaussianNB classifier...")
    clf = GaussianNB()
    clf.fit(train_embeddings, y_train)
    
    print(f"Saving classifier to {CLASSIFIER_SAVE_PATH}...")
    joblib.dump(clf, CLASSIFIER_SAVE_PATH)
    print("Export complete.")

if __name__ == "__main__":
    main()
