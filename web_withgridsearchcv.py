
# data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import streamlit as st
# import os 
# os.system('pip install scikit-learn')

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords.ensure_loaded()
from nltk.stem import PorterStemmer
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler 
# Pipeline and model
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
# Score of models
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, f1_score, recall_score, balanced_accuracy_score

dfblacklist = pd.read_csv('spam_words.txt', header=None, on_bad_lines='skip' )
dfblacklist.rename(columns={0:'words'}, inplace=True)
dfblacklistList = dfblacklist['words'].tolist()

def cree_df(url = "SMSSpamCollection.txt"):
    df = pd.read_csv(url, sep='\t', header=None )
    df.rename(columns={0:'type',1:'mail'}, inplace=True)
    return df

def prep(df): 
    df['minuscule']=df['mail'].str.lower()
    tokenizer = RegexpTokenizer(r"\b\w+\b|\d{2} \d{2} \d{2} \d{2} \d{2}")
    df['token'] = df['minuscule'].apply(lambda x: tokenizer.tokenize(x))
    stop = stopwords.words('english')
    df['without_stopwords']=df['token'].apply(lambda x: [word for word in x if word not in stop])
    stemmer = PorterStemmer()
    df['PorterStemmer'] = df['without_stopwords'].apply(lambda x: [stemmer.stem(word) for word in x])
    df['clean'] = df['without_stopwords'].apply(lambda x: " ".join(x))
    return df

def features(df):
    df['len']=df['mail'].str.len()
# df['nombre_mots']=df['mail'].str.split().str.len()
    df['nombre_mots']=df['token'].str.len()
    pattern = r"http\S+|www.\S+"
    df['http']=df['mail'].apply(lambda x : True if re.search(pattern, x) else False)

    pattern = r"/^[\(]?[\+]?(\d{2}|\d{3})[\)]?[\s]?((\d{6}|\d{8})|(\d{3}[\*\.\-\s]){3}|(\d{2}[\*\.\-\s]){4}|(\d{4}[\*\.\-\s]){2})|\d{8}|\d{10}|\d{12}$/"

    df['phone']=df['mail'].apply(lambda x : True if re.search(pattern, x) else False)

    #pattern = r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    #df['mail_compt']=df['mail'].apply(lambda x: re.search(pattern, x))

    df['blacklist']=df['token'].apply(lambda x: len([ word for word in x if word  in dfblacklistList]))
    return df


def spliteur_simple(df):
    X = df.drop(columns = ['type'], axis=1)
    y = df['type']
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def spliteur(df):
    X = df.drop(columns = ['type'], axis=1)
    y = df['type']
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    #return train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    return train_test_split(X_res, y_res, stratify=y_res, test_size=0.2, random_state=42)

def ModelCreateur(X_train, y_train, classifier):

    column_num  = ['len','nombre_mots','blacklist']
    column_bool = ['http','phone']
    
    #Transformation des variables texte
    transfo_text_TFid = Pipeline(steps=[
        ('Tfid', TfidfVectorizer(lowercase=False, decode_error='ignore', analyzer='char_wb', ngram_range=(2, 2)))
        
    ])

    #Application des étapes sur tout le dataset
    if isinstance(classifier, ComplementNB) or isinstance(classifier, MultinomialNB):
        preparation = ColumnTransformer(
        transformers=[
        ('TFid&data', transfo_text_TFid , 'clean'), #TFIDF ne prend pas de listes comme arguments
        # ('CountVect&data', transfo_text_CountVect , 'clean'),
            ('Scaler&data',MinMaxScaler(), column_num),
            ('BoolEncoder',OrdinalEncoder(), column_bool)
        ])
    else : 
        preparation = ColumnTransformer(
        transformers=[
        ('TFid&data', transfo_text_TFid , 'clean'), #TFIDF ne prend pas de listes comme arguments
        # ('CountVect&data', transfo_text_CountVect , 'clean'),
            ('Scaler&data',RobustScaler(), column_num),
            ('BoolEncoder',OrdinalEncoder(), column_bool)
        ])
    
    #relie l'algorithme avec le modèle
    model = Pipeline([
    ('vectorizer', preparation),
    ('model', classifier)
    ])
    #Fit le modèle
    model.fit(X_train, y_train)
    return model

def AfficherScores(y_test, y_pred):
    
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.dataframe(classification_report(y_test, y_pred, output_dict=True))
    
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    
    # plt.hist(model_lm.decision_function(X_test), bins=50)
    plt.show()

# fonction qui affiche la matrice de confusion du modèle
def matrixconf(y_test,y_pred):
    #affiche la matrice de confusion du modèle
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

def testModel(sms,model):
  
    df_sms = pd.DataFrame({'type': ['test'], 'mail': [sms]})
    df_sms        = prep(df_sms)
    df_sms        = features(df_sms)
    result = model.predict(df_sms)
    return result


# def modellist():
#     classifier1 = LogisticRegression(solver='lbfgs,liblinear', C=1e3)
#     classifier2 = KNeighborsClassifier(7)

#     classifier3 = ComplementNB()                                        #0.9838516746411483
#     classifier4 = MultinomialNB()                                       #0.9856459330143541
#     classifier5 = BernoulliNB(force_alpha=True)

#     classifier6 = svm.SVC()                                                #0.9742822966507177
#     classifier7 = SVC(gamma=2, C=1, random_state=42)                    #0.8941387559808612

#     classifier8 = RidgeClassifier(tol=1e-2, solver="sparse_cg")          #0.9811659192825112
#     classifier9 = RandomForestClassifier(max_depth=200, random_state=42) #0.9838516746411483
#     classifier10 = DecisionTreeClassifier()                              #0.9700956937799043

#     return [classifier1,classifier2,classifier3,classifier4,classifier5,classifier6,classifier7,classifier8,classifier9,classifier10]

# def model_parametre():
#     parametre1 = {'model__solver':'liblinear', 'C':[1e3, 1e4]}
#     parametre2 = KNeighborsClassifier(7)

#     parametre3 = ComplementNB()                                       
#     parametre4 = MultinomialNB()                                       
#     parametre5 = BernoulliNB(force_alpha=True)

#     parametre6 = {'model__kernel':('linear', 'rbf'), 'model__C':[1, 10]}
#     parametre7 = SVC(gamma=2, C=1, random_state=42)                    

#     parametre8 = RidgeClassifier(tol=1e-2, solver="sparse_cg")          
#     parametre9 = RandomForestClassifier(max_depth=200, random_state=42)
#     parametre10 = DecisionTreeClassifier()                              

#     return [parametre1,parametre2,parametre3,parametre4,parametre5,parametre6,parametre7,parametre8,parametre9,parametre10]



def gridCreateur(pipe, parametre):
    Scoring_list= ['accuracy', 'recall', 'roc_auc']

    # grid = GridSearchCV(pipe, parametre, scoring=Scoring_list, refit='roc_auc', cv = 5, n_jobs =-1, verbose = 1)
    grid = GridSearchCV(pipe, parametre, scoring='accuracy', cv = 5, n_jobs =-1, verbose = 1)
    # Fit the model
    grid.fit(X_train, y_train)
    return grid



def modeleslist():
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(),'param': {'model__solver':['liblinear'], 'model__C':[0.1, 1, 10]}
        },
        'SVC': {
            'model': SVC(),'param': {'model__kernel':('linear', 'rbf'), 'model__C':[1, 10]}
        }
    }
    return models

    
    

# """"""""""""""""Main""""""""""""""""""

dfModel = cree_df("SMSSpamCollection.txt")
dfModel = prep(dfModel)
dfModel = features(dfModel)    
X_train, X_test, y_train, y_test = spliteur_simple(dfModel)

list_model = modeleslist()
st.image('logoweb.png', use_column_width="auto")
st.title('SPAM DETECTOR 🔎')
input =  st.text_input('Enter a message')
option = st.selectbox('Sélectionnez un modele', list_model.keys())
submit = st.button('Predict')

if submit:
    model_lm = ModelCreateur(X_train, y_train, list_model[option]['model'])
    model=gridCreateur(model_lm,list_model[option]['param'])

    y = testModel(input,model)
    if y == 'spam':
        st.write("it's a spam")
    else:
        st.write("it's not a spam")
    
    y_pred = model.predict(X_test)

    AfficherScores(y_test, y_pred)




# if submit:
#     for i in list_model:
#         model_lm = ModelCreateur(X_train, y_train, i)
#         y_pred = testModel(str(input),model_lm)
#         if y_pred == 'spam':
#             st.write(i,"it's a spam")
#         else:
#             st.write(i,"it's not a spam")


