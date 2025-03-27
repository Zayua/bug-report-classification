import os
import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix, classification_report)

# Classical ML Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Text cleaning & stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

########## 1. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Complete text preprocessing pipeline."""
    # Remove HTML tags
    text = remove_html(text)
    # Remove emojis
    text = remove_emoji(text)
    # Clean text by removing non-alphanumeric characters and convert to lowercase
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = text.strip().lower()
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in final_stop_words_list]
    
    return " ".join(words)

########## 2. Feature Engineering Functions ##########

# Function to create embeddings for deep learning models
def create_embeddings(texts, max_features=10000, maxlen=100):
    """Create word embeddings for deep learning models."""
    # Tokenize texts
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    
    return padded_sequences, tokenizer

# Function to create TF-IDF features
def create_tfidf_features(train_texts, test_texts, max_features=1000, ngram_range=(1, 2)):
    """Create TF-IDF features."""
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )
    X_train = tfidf.fit_transform(train_texts).toarray()
    X_test = tfidf.transform(test_texts).toarray()
    
    return X_train, X_test, tfidf

########## 3. Model Definitions ##########

# Function to build LSTM model
def build_lstm_model(max_features, maxlen, embedding_dim=128):
    """Build an LSTM model for sentiment analysis."""
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=maxlen),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

# Function to build Bidirectional LSTM model
def build_bidirectional_lstm_model(max_features, maxlen, embedding_dim=128):
    """Build a Bidirectional LSTM model for sentiment analysis."""
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=maxlen),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

# Function to build GRU model
def build_gru_model(max_features, maxlen, embedding_dim=128):
    """Build a GRU model for sentiment analysis."""
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=maxlen),
        GRU(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

########## 4. Evaluation Functions ##########

def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """Evaluate model and return metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Compute AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_val = auc(fpr, tpr)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc_val
    }

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{output_dir}/{model_name}_confusion_matrix.png')
    plt.close()

########## 5. Model Configuration ##########

# Number of repeated experiments
REPEAT = 5

# List of models to evaluate
def get_models_to_evaluate():
    return [
        {
            'name': 'Naive_Bayes',
            'model': GaussianNB(),
            'params': {'var_smoothing': np.logspace(-12, 0, 13)},
            'type': 'classical'
        },
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'type': 'classical'
        },
        {
            'name': 'SVM',
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'type': 'classical'
        },
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'type': 'classical'
        },
        {
            'name': 'LSTM',
            'build_fn': build_lstm_model,
            'type': 'deep_learning'
        },
        {
            'name': 'BiLSTM',
            'build_fn': build_bidirectional_lstm_model,
            'type': 'deep_learning'
        },
        {
            'name': 'GRU',
            'build_fn': build_gru_model,
            'type': 'deep_learning'
        }
    ]

########## 6. Ensemble Model Function ##########

def train_ensemble_model(data, text_col, model_configs, output_dir, n_splits=5):
    """Train an ensemble of models using cross-validation."""
    
    # Create a stratified k-fold cross-validator
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lists to store fold results
    fold_results = []
    
    # Preprocess all text data once
    X = data[text_col]
    y = data['sentiment']
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\nTraining ensemble - Fold {fold+1}/{n_splits}")
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        # Dictionary to store predictions from each model
        model_predictions = {}
        
        # Train each model and collect predictions
        for model_config in model_configs:
            model_name = model_config['name']
            model_type = model_config['type']
            
            print(f"Training {model_name} for ensemble...")
            
            if model_type == 'classical':
                # Create TF-IDF features
                X_train_tfidf, X_test_tfidf, _ = create_tfidf_features(
                    X_train, X_test, max_features=2000, ngram_range=(1, 3)
                )
                
                # Train model with best hyperparameters
                model = model_config['model']
                model.fit(X_train_tfidf, y_train)
                
                # Make predictions
                pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
                model_predictions[model_name] = pred_proba
                
            elif model_type == 'deep_learning':
                # Create embeddings
                max_features = 10000
                maxlen = 100
                
                X_train_emb, tokenizer = create_embeddings(X_train, max_features, maxlen)
                X_test_emb = tokenizer.texts_to_sequences(X_test)
                X_test_emb = pad_sequences(X_test_emb, maxlen=maxlen)
                
                # Build and train model
                model = model_config['build_fn'](max_features, maxlen)
                
                model.fit(
                    X_train_emb, y_train,
                    epochs=5,  # Reduce epochs for faster training
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Make predictions
                pred_proba = model.predict(X_test_emb).flatten()
                model_predictions[model_name] = pred_proba
        
        # Create ensemble prediction (average of all model predictions)
        ensemble_pred_proba = np.mean(list(model_predictions.values()), axis=0)
        ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
        
        # Evaluate ensemble
        metrics = evaluate_model(y_test, ensemble_pred_proba)
        fold_results.append(metrics)
        
        print(f"Fold {fold+1} - Ensemble Results:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")
    
    # Calculate average metrics across folds
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_results])
        for metric in fold_results[0].keys()
    }
    
    return avg_metrics

########## 7. Main Project Analysis Function ##########

def run_project_analysis(project):
    """
    Run sentiment analysis for a specific project.
    
    Args:
        project (str): Name of the project to analyze
    """
    # Reset random seeds for each project
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print(f"\n{'='*50}")
    print(f"Analyzing Project: {project.upper()}")
    print(f"{'='*50}")
    
    # Path configuration
    path = f'./datasets/{project}.csv'
    output_dir = f'./model_results/{project}'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Loading dataset from {path}...")
        pd_all = pd.read_csv(path)
        pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

        # Merge Title and Body into a single column; if Body is NaN, use Title only
        pd_all['Title+Body'] = pd_all.apply(
            lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
            axis=1
        )

        # Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
        pd_tplusb = pd_all.rename(columns={
            "Unnamed: 0": "id",
            "class": "sentiment",
            "Title+Body": "text"
        })
        
        # Save project-specific intermediate file
        intermediate_file = f'{output_dir}/{project}_Title+Body.csv'
        pd_tplusb.to_csv(intermediate_file, index=False, columns=["id", "Number", "sentiment", "text"])

        # Read the saved data
        data = pd.read_csv(intermediate_file).fillna('')
        text_col = 'text'

        # Keep a copy for referencing original data if needed
        original_data = data.copy()

        # Apply text preprocessing
        print("Preprocessing text data...")
        data[text_col] = data[text_col].apply(preprocess_text)

        # Display dataset info
        print(f"Dataset shape: {data.shape}")
        print(f"Sentiment distribution:\n{data['sentiment'].value_counts()}")

        # Number of repeated experiments
        REPEAT = 5

        # Dictionary to store results for each model
        all_results = {}

        # Get models to evaluate
        models_to_evaluate = get_models_to_evaluate()

        # Training and Evaluation Loop
        for model_config in models_to_evaluate:
            model_name = model_config['name']
            model_type = model_config['type']
            
            print(f"\n{'='*50}")
            print(f"Training {model_name} model...")
            print(f"{'='*50}")
            
            # Lists to store metrics across repeated runs
            metrics_list = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'auc': []
            }
            
            for rep in range(REPEAT):
                print(f"\nRepetition {rep+1}/{REPEAT}")
                
                # Split data into train and test sets
                indices = np.arange(data.shape[0])
                train_index, test_index = train_test_split(
                    indices, test_size=0.2, random_state=rep*42
                )
                
                train_text = data[text_col].iloc[train_index]
                test_text = data[text_col].iloc[test_index]
                
                y_train = data['sentiment'].iloc[train_index]
                y_test = data['sentiment'].iloc[test_index]
                
                if model_type == 'classical':
                    # Create TF-IDF features for classical ML models
                    X_train, X_test, _ = create_tfidf_features(
                        train_text, test_text, max_features=2000, ngram_range=(1, 3)
                    )
                    
                    # Grid search for hyperparameter tuning
                    grid = GridSearchCV(
                        model_config['model'],
                        model_config['params'],
                        cv=5,
                        scoring='roc_auc',
                        n_jobs=-1
                    )
                    
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    
                    # Make predictions
                    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                    y_pred = best_model.predict(X_test)
                    
                    # Evaluate model
                    metrics = evaluate_model(y_test, y_pred_proba)
                    
                    print(f"Best parameters: {grid.best_params_}")
                    
                elif model_type == 'deep_learning':
                    # Create embeddings for deep learning models
                    max_features = 10000
                    maxlen = 100
                    
                    X_train, tokenizer = create_embeddings(train_text, max_features, maxlen)
                    X_test = tokenizer.texts_to_sequences(test_text)
                    X_test = pad_sequences(X_test, maxlen=maxlen)
                    
                    # Build model
                    model = model_config['build_fn'](max_features, maxlen)
                    
                    # Define callbacks
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                        ModelCheckpoint(
                            f'{output_dir}/{model_name}_model_{rep}.h5',
                            monitor='val_loss',
                            save_best_only=True
                        )
                    ]
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Make predictions
                    y_pred_proba = model.predict(X_test).flatten()
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    
                    # Evaluate model
                    metrics = evaluate_model(y_test, y_pred_proba)
                    
                    # Plot training history
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['accuracy'])
                    plt.plot(history.history['val_accuracy'])
                    plt.title(f'{model_name} - Accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Validation'], loc='upper left')
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title(f'{model_name} - Loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Validation'], loc='upper left')
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/{model_name}_history_{rep}.png')
                    plt.close()
                
                # Store metrics
                for metric, value in metrics.items():
                    metrics_list[metric].append(value)
                
                # Plot confusion matrix for the last repetition
                if rep == REPEAT - 1:
                    plot_confusion_matrix(y_test, y_pred, model_name, output_dir)
                    
                    # Print classification report
                    print(f"\nClassification Report for {model_name}:")
                    print(classification_report(y_test, y_pred))
            
            # Calculate average metrics
            avg_metrics = {metric: np.mean(values) for metric, values in metrics_list.items()}
            
            # Print and store results
            print(f"\n=== {model_name} Results ===")
            print(f"Average Accuracy:  {avg_metrics['accuracy']:.4f}")
            print(f"Average Precision: {avg_metrics['precision']:.4f}")
            print(f"Average Recall:    {avg_metrics['recall']:.4f}")
            print(f"Average F1 Score:  {avg_metrics['f1']:.4f}")
            print(f"Average AUC:       {avg_metrics['auc']:.4f}")
            
            # Store results for model comparison
            all_results[model_name] = avg_metrics

        # Select top models for ensemble
        top_models = [
            next(config for config in models_to_evaluate if config['name'] == 'LogisticRegression'),
            next(config for config in models_to_evaluate if config['name'] == 'SVM'),
            next(config for config in models_to_evaluate if config['name'] == 'BiLSTM')
        ]

        # Train and evaluate ensemble
        ensemble_results = train_ensemble_model(data, text_col, top_models, output_dir, n_splits=3)

        # Print ensemble results
        print("\n=== Ensemble Model Results ===")
        print(f"Average Accuracy:  {ensemble_results['accuracy']:.4f}")
        print(f"Average Precision: {ensemble_results['precision']:.4f}")
        print(f"Average Recall:    {ensemble_results['recall']:.4f}")
        print(f"Average F1 Score:  {ensemble_results['f1']:.4f}")
        print(f"Average AUC:       {ensemble_results['auc']:.4f}")

        # Add ensemble results to the comparison
        all_results['Ensemble'] = ensemble_results

        # Create a DataFrame for easy comparison
        results_df = pd.DataFrame(all_results).T
        results_df.index.name = 'Model'
        results_df = results_df.reset_index()

        # Save results to CSV
        results_df.to_csv(f'{output_dir}/{project}_final_comparison.csv', index=False)

        # Final comparison plot
        plt.figure(figsize=(12, 8))

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            sns.barplot(x='Model', y=metric, data=results_df)
            plt.title(f'Comparison of {metric.capitalize()}')
            plt.xticks(rotation=45)
            plt.tight_layout()

        plt.savefig(f'{output_dir}/{project}_final_comparison.png')
        plt.close()

        print(f"\nResults summary:")
        print(results_df.to_string(index=False))
        print(f"\nAll results have been saved to {output_dir}/")

    except Exception as e:
        print(f"Error processing project {project}: {e}")
        import traceback
        traceback.print_exc()

########## 8. Main Execution ##########

def main():
    # Define the projects to analyze
    PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
    os.makedirs('./model_results', exist_ok=True)
    # Analyze each project
    for project in PROJECTS:

        os.makedirs(f'./model_results/{project}', exist_ok=True)
        run_project_analysis(project)

if __name__ == "__main__":
    main()