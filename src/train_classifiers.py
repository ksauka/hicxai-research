import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import preprocess_adult
from load_adult_data import load_adult_data


def train_and_evaluate(X, y, model, model_name, models_dir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # Save model
    joblib.dump(model, os.path.join(models_dir, f'{model_name}.pkl'))

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    df, _ = load_adult_data(data_dir)
    df_clean = preprocess_adult(df)
    X = df_clean.drop('income', axis=1)
    y = df_clean['income']

    classifiers = [
        (RandomForestClassifier(n_estimators=100, random_state=42), 'RandomForest'),
        (GradientBoostingClassifier(n_estimators=100, random_state=42), 'GradientBoosting'),
        (AdaBoostClassifier(n_estimators=100, random_state=42), 'AdaBoost'),
        (SVC(kernel='rbf', probability=True, random_state=42), 'SVM'),
        (LogisticRegression(max_iter=1000, random_state=42), 'LogisticRegression')
    ]

    for clf, name in classifiers:
        train_and_evaluate(X, y, clf, name, models_dir)
