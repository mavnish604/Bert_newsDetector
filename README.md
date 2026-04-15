# Fake News Detection Project Report

## Project Overview
The Fake News Detection project aims to classify news articles as either "Fake" or "True" using machine learning techniques. The project leverages the WELFake dataset, which contains labeled news articles, and employs advanced natural language processing (NLP) models to generate embeddings for classification.

## Dataset
The dataset used in this project is the WELFake dataset, which consists of three CSV files:
- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains true news articles.
- `WELFake_Dataset.csv`: A combined dataset used for training and evaluation.

### Data Preprocessing
1. The dataset was loaded using pandas.
2. Missing values were identified and dropped.
3. Labels were inverted to ensure consistency (1 for True, 0 for Fake).
4. The dataset was split into training and testing sets using an 80-20 ratio.

## Model Training
### Sentence Embeddings
The project uses the `SentenceTransformer` model (`all-MiniLM-L6-v2`) to generate sentence embeddings for the news articles. These embeddings capture the semantic meaning of the text and are used as input features for classification.

### Training Details
1. **Loss Function**: Batch Hard Triplet Loss was used to train the SentenceTransformer model.
2. **Training Arguments**:
   - Number of epochs: 3
   - Batch size: 16
   - Evaluation strategy: Per epoch
   - Save strategy: Per epoch
3. **Trainer**: The `SentenceTransformerTrainer` was used to train the model on the processed dataset.

## Evaluation
### Logistic Regression Classifier
A logistic regression model was trained on the sentence embeddings generated for the training set. The model was evaluated on the test set.

### Metrics
1. **Confusion Matrix**: Visualized to show the classification performance.
2. **Classification Report**: Includes precision, recall, F1-score, and accuracy for both classes (Fake and True).

### Results
The logistic regression classifier achieved the following results:
- High accuracy in distinguishing between fake and true news articles.
- Detailed metrics are provided in the classification report.

### PCA Visualization
Principal Component Analysis (PCA) was applied to reduce the dimensionality of the sentence embeddings to 2D. A scatter plot was generated to visualize the separation between fake and true news articles.

## Tools and Libraries
The following tools and libraries were used in this project:
- Python
- KaggleHub
- Pandas
- Scikit-learn
- SentenceTransformers
- Datasets
- Matplotlib
- Seaborn

## Directory Structure
```
.
├── app.py
├── classifier.joblib
├── export_classifier.py
├── LICENSE
├── README.md
├── requirements.txt
├── Untitled.ipynb
├── WELFake.ipynb
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── WELFake_Dataset.csv
├── fake_news_model/
│   ├── checkpoint-10731/
│   ├── checkpoint-2245/
│   ├── checkpoint-3577/
│   ├── checkpoint-4490/
│   ├── checkpoint-6735/
│   └── checkpoint-7154/
```

## Conclusion
The Fake News Detection project successfully demonstrates the use of NLP techniques and machine learning models to classify news articles. The combination of SentenceTransformer embeddings and logistic regression provides a robust solution for this classification task. Future work could explore:
- Fine-tuning the SentenceTransformer model on the WELFake dataset.
- Experimenting with other classification algorithms.
- Deploying the model as a web application for real-time fake news detection.