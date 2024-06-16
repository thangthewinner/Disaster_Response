import sys
import pandas as pd
import os
import re
import urllib
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine

def load_data(database_filepath=None):
    params = urllib.parse.quote_plus('DRIVER={ODBC Driver 18 for SQL Server};SERVER=adyserver2.database.windows.net;DATABASE=ADY201m_Lab04;\
                                    UID=admin99;PWD=Matkhau@123')

    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    df = pd.read_sql_table('tblDisaster', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X,y,category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model(clf = AdaBoostClassifier()):
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    parameters = {
        'clf__estimator__learning_rate':[0.5, 1.0],
        'clf__estimator__n_estimators':[10,20]
    
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)



def main():
    if len(sys.argv) == 2:
        model_filepath = sys.argv[1] 
        print('Loading data...\n    DATABASE: {}')
        X, Y, category_names = load_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py classifier.pkl')



if __name__ == '__main__':
    main()