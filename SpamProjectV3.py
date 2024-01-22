
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
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from sklearn.preprocessing import OrdinalEncoder

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
# Score of models
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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

    #pattern = r"[-A-Za-z0-9!#$%&'*+/=?^_`{|}~]+(?:\.[-A-Za-z0-9!#$%&'*+/=?^_`{|}~]+)*@(?:[A-Za-z0-9](?:[-A-Za-z0-9]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[-A-Za-z0-9]*[A-Za-z0-9])?"
    #df['mail_compt']=df['mail'].apply(lambda x: re.search(pattern, x))

    df['blacklist']=df['token'].apply(lambda x: len([ word for word in x if word  in dfblacklistList]))
    return df


def spliteur(df):
    X = df.drop(columns = ['type'], axis=1)
    y = df['type']
    return train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

def ModelCreateur(X_train, y_train, classifier):

    column_num  = ['len','nombre_mots','blacklist']
    column_bool = ['http','phone']
    
    #Transformation des variables texte
    transfo_text_TFid = Pipeline(steps=[
        ('Tfid', TfidfVectorizer(lowercase=False, decode_error='ignore', analyzer='char_wb', ngram_range=(2, 2)))
        
    ])

    #Application des étapes sur tout le dataset
    if classifier == "ComplementNB()" or "MultinomialNB()":
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
    ('classifier', classifier)
    ])
    #Fit le modèle
    model.fit(X_train, y_train)
    return model

def AfficherScores(y_test, y_pred):
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    
    # plt.hist(model_lm.decision_function(X_test), bins=50)
    plt.show()

# fonction qui affiche la matrice de confusion du modèle
def matrixconf(y_test,y_pred):
    #affiche la matrice de confusion du modèle
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

def testModel(sms,model):
    input_sms     = sms
    df_sms        = pd.DataFrame(input_sms)
    df_sms.rename(columns={0:'mail'}, inplace=True)
    df_sms        = prep(df_sms)
    df_sms        = features(df_sms)


    result = model.predict(df_sms)
    return result

dfModel = cree_df("SMSSpamCollection.txt")
dfModel = prep(dfModel)
dfModel = features(dfModel)
X_train, X_test, y_train, y_test = spliteur(dfModel)

classifier1 = LogisticRegression(solver='liblinear', C=1e3)
classifier2 = KNeighborsClassifier(7)

classifier3 = ComplementNB()                                        #0.9838516746411483
classifier4 = MultinomialNB()                                       #0.9856459330143541
classifier5 = BernoulliNB(force_alpha=True)

classifier6 = SVC(gamma='auto')                                     #0.9742822966507177
classifier7 = SVC(gamma=2, C=1, random_state=42)                    #0.8941387559808612

classifier8 = RidgeClassifier(tol=1e-2, solver="sparse_cg")          #0.9811659192825112
classifier9 = RandomForestClassifier(max_depth=200, random_state=42) #0.9838516746411483
classifier10 = DecisionTreeClassifier()                              #0.9700956937799043

list_model = [classifier1,classifier2,classifier3,classifier4,classifier5,classifier6,classifier7,classifier8,classifier9,classifier10]




input =  ['Hi Nick. This is to remind you about the $75 minimum payment on your credit card ending in XXXX. Payment is due on 01/01. Pls visit order.com to make your payment']
for i in list_model:
    model_lm=ModelCreateur(X_train, y_train, i)
    # print(i,':',testModel(model_lm))
    print('model utilisé:', i)
    y_pred = model_lm.predict(X_test)
    AfficherScores(y_test, y_pred)
    matrixconf(y_test,y_pred)

    model_disp = RocCurveDisplay.from_estimator(model_lm,X_test,y_test)



