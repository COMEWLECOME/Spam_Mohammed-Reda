# data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import streamlit as st
# Preprocessing
from imblearn.under_sampling import RandomUnderSampler #conda install conda-forge::imbalanced-learn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
# nltk.download('punkt')
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

# Pipeline and model
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
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# Score of models
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_score

dfblacklist = pd.read_csv('spam_words.txt', header=None, on_bad_lines='skip' )
dfblacklist.rename(columns={0:'words'}, inplace=True)
dfblacklistList = dfblacklist['words'].tolist()

def cree_df(url = "SMSSpamCollection.txt"):
    df = pd.read_csv(url, sep='\t', header=None )
    df.rename(columns={0:'type',1:'mail'}, inplace=True)
    return df

#fonction de préparation des données (preprocessing)
def prep(df): 

    #lowercase des message
    df['minuscule']=df['mail'].str.lower()

    #mise en place des tokens des message
    tokenizer = RegexpTokenizer(r"\b\w+\b|\d{2} \d{2} \d{2} \d{2} \d{2}")
    df['token'] = df['minuscule'].apply(lambda x: tokenizer.tokenize(x))

    #ajout d'un stopwords 
    stop = stopwords.words('english')
    df['without_stopwords']=df['token'].apply(lambda x: [word for word in x if word not in stop])

    #ajout d'un stemmer
    stemmer = PorterStemmer()
    df['PorterStemmer'] = df['without_stopwords'].apply(lambda x: [stemmer.stem(word) for word in x])
    
    #regroupement du traitement des données
    df['clean'] = df['without_stopwords'].apply(lambda x: " ".join(x))
    return df

def features(df):
    
    #ajout d'une feature "longueur du message"
    df['len']=df['mail'].str.len()

    #ajout d'une feature "nombre de mots"
# df['nombre_mots']=df['mail'].str.split().str.len()
    df['nombre_mots']=df['token'].str.len()

    #ajout d'une feature permettant de vérifier si présence d'hypertexte
    pattern = r"http\S+|www.\S+"
    df['http']=df['mail'].apply(lambda x : True if re.search(pattern, x) else False)

    #ajout d'une feature permettant de vérifier la présence de chiffre 
    pattern = r"/^[\(]?[\+]?(\d{2}|\d{3})[\)]?[\s]?((\d{6}|\d{8})|(\d{3}[\*\.\-\s]){3}|(\d{2}[\*\.\-\s]){4}|(\d{4}[\*\.\-\s]){2})|\d{8}|\d{10}|\d{12}$/"
    df['phone']=df['mail'].apply(lambda x : True if re.search(pattern, x) else False)
    
    #ajout d'une feature permettant de vérifier la présence de mail
    pattern = r"[-A-Za-z0-9!#$%&'*+/=?^_`{|}~]+(?:\.[-A-Za-z0-9!#$%&'*+/=?^_`{|}~]+)*@(?:[A-Za-z0-9](?:[-A-Za-z0-9]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[-A-Za-z0-9]*[A-Za-z0-9])?"
    df['mail_compt']=df['mail'].apply(lambda x : True if re.search(pattern, x) else False)

    #ajout d'une feature permettant de vérifier la présence de mots blacklisté 
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

def SMOTE_simple(df):
    X = df[['clean','len','nombre_mots','blacklist','http','phone','mail_compt']]
    y = df['type']
    rus = SMOTENC(random_state=42, categorical_features=[0])
    X_res, y_res = rus.fit_resample(X, y)
    #return train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    return train_test_split(X_res, y_res, stratify=y_res, test_size=0.2, random_state=42)

def ModelCreateur(X_train, y_train, classifier):

    column_num  = ['len','nombre_mots','blacklist']
    column_bool = ['http','phone','mail_compt']
    column_text = 'clean'
    #Transformation des variables texte
    transfo_text_TFid = Pipeline(steps=[
        ('Tfid', TfidfVectorizer(lowercase=False, decode_error='ignore', analyzer='char_wb', ngram_range=(2, 2)))
        
    ])

#Application des étapes sur tout le dataset
    if isinstance(classifier, ComplementNB) or isinstance(classifier, MultinomialNB) or isinstance(classifier, BernoulliNB):
        preparation = ColumnTransformer(
        transformers=[
        ('TFid&data', transfo_text_TFid , column_text), #TFIDF ne prend pas de listes comme arguments
        # ('CountVect&data', transfo_text_CountVect , column_text),
            ('Scaler&data',MinMaxScaler(), column_num),
            ('BoolEncoder',OrdinalEncoder(), column_bool)
        ])
    else : 
        preparation = ColumnTransformer(
        transformers=[
        ('TFid&data', transfo_text_TFid , column_text), #TFIDF ne prend pas de listes comme arguments
        # ('CountVect&data', transfo_text_CountVect , column_text),
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

# def AfficherScores(y_test, y_pred):
    
#     st.write("Accuracy:", accuracy_score(y_test, y_pred))
#     st.dataframe(classification_report(y_test, y_pred, output_dict=True))
    
#     ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    
#     plt.hist(model_lm.decision_function(X_test), bins=50)
#     plt.show()

#fonction permettant de connaître le score de notre modèle
def AfficherScores(model,y_test, y_pred):
    
    classifier_name = model.best_estimator_.named_steps['model'].__class__.__name__
    st.write(f"--------------------------------{classifier_name}-------------------------------------------------")
    
    #affiche la classification rapport du modèle
    st.dataframe(classification_report(y_test, y_pred, output_dict=True))
    st.write(f"Best parameters: {model.best_params_}")
    st.write(f"Best score: {model.best_score_}")
    
    
    disp = matrixconf(y_test,y_pred)
    plt.title(f"Matrice de confusion de {classifier_name} ")
    st.pyplot(plt)
    disp = RocCurveDisplay.from_estimator(model,X_test,y_test)
    plt.title(f"Courbe ROC de {classifier_name} ")
    st.pyplot(plt)



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

def gridCreateur(pipe, parametre):
    
    # grid = GridSearchCV(pipe, parametre, scoring=Scoring_list, refit='roc_auc', cv = 5, n_jobs =-1, verbose = 1)
    R = make_scorer(recall_score, pos_label='spam')
    P = make_scorer(precision_score, pos_label='spam')
    Scoring_list= {'Myrecall':R, 'MyPrecision':P, 'roc':'roc_auc'}
    grid = GridSearchCV(pipe, parametre, scoring=Scoring_list, refit='Myrecall',cv = 2, n_jobs =-1, verbose = 1)
    # Fit the model
    grid.fit(X_train, y_train)
    return grid



def modeleslist():
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(),'param': {'model__solver':['liblinear'],
                                                    'model__penalty': ['l1'],
                                                    'model__C':[10]}
        },
        'ComplementNB': {
            'model': ComplementNB(),'param': {'model__alpha': [0.1]}
        },
        'BernoulliNB': {
            'model': BernoulliNB(),'param': {'model__alpha': [0.0],
                                              'model__binarize': [None]}
        },        
        'SVC': {
            'model': SVC(),'param': {'model__kernel':['rbf'],
                                     'model__C':[10]}
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),'param': {'model__n_estimators': [200],
                                                        'model__max_depth': [30]}
        
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(),'param': {'model__criterion': ['gini'],
                                                        'model__splitter': ['best'],
                                                        'model__max_depth': [30]
                                                        }
        
        },
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier(),'param': {'model__n_neighbors': [3],
                                                      'model__weights': ['distance'],
                                                    'model__algorithm': ['auto']}
        
        },
        'RidgeClassifier': {
            'model': RidgeClassifier(),'param': {'model__alpha': [0.1],
                                                 'model__solver': ['auto']
                                                 }
        }
    }
    return models

    
    

# """"""""""""""""Main""""""""""""""""""

dfModel = cree_df("SMSSpamCollection.txt")
dfModel = prep(dfModel)
dfModel = features(dfModel)    
X_train, X_test, y_train, y_test = SMOTE_simple(dfModel)

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

    AfficherScores(model,y_test, y_pred)




# if submit:
#     for i in list_model:
#         model_lm = ModelCreateur(X_train, y_train, i)
#         y_pred = testModel(str(input),model_lm)
#         if y_pred == 'spam':
#             st.write(i,"it's a spam")
#         else:
#             st.write(i,"it's not a spam")


