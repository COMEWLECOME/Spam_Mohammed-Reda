
# data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
# Pipeline and model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
# Score of models
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report, confusion_matrix

def cree_df(url = "SMSSpamCollection.txt"):
    df = pd.read_csv(url, sep='\t', header=None )
    df.rename(columns={0:'type',1:'mail'}, inplace=True)
    return df

def spliteur(df):
    X = df.drop(['type'], axis=1)
    y = df['type']
    return train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

def PipeCreateur(X_train, y_train, classifier):
    
    column_text = X_train.select_dtypes(include=['object']).columns
    
    # Transformation of textual variables
    transfo_text_TFid = Pipeline(steps=[
        ('Tfid', TfidfVectorizer(lowercase=False, decode_error='ignore', analyzer='char_wb', ngram_range=(2, 2)))
        
    ])

    # Class ColumnTransformer : apply alls steps on the whole dataset
    preparation = ColumnTransformer(
        transformers=[
            ('TFid&data', transfo_text_TFid , column_text[0]), #TFIDF ne prend pas de listes comme arguments
        ])
    
    model_lm = Pipeline([
    ('vectorizer', preparation),
    ('classifier', classifier)
    ])
    # Fit the model
    model_lm.fit(X_train, y_train)
    return model_lm

def AfficherScores(y_test, y_pred):
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    
    # plt.hist(model_lm.decision_function(X_test), bins=50)
    plt.show()
    
    
X_train, X_test, y_train, y_test = spliteur(cree_df("SMSSpamCollection.txt"))

classifier =LogisticRegression(solver='liblinear', C=1e3)

model_lm=PipeCreateur(X_train, y_train, classifier)

y_pred = model_lm.predict(X_test)
AfficherScores(y_test, y_pred)