# %%
# Load data
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# Pipeline and model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report
# Score of models
from sklearn.metrics import accuracy_score

# tokenizer avec RE (regular expressions)
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer


# %%
url = "SMSSpamCollection.txt"
df = pd.read_csv(url, sep='\t', header=None )

df.rename(columns={0:'type',1:'mail'}, inplace=True)

# %%
df['minuscule']=df['mail'].str.lower()

# %%
# tokenizer = RegexpTokenizer(r"[a-zA-Z]\w+\'?\w*")
tokenizer = RegexpTokenizer(r"\b\w+\b|\d{2} \d{2} \d{2} \d{2} \d{2}|http\S+")

df['token'] = df['minuscule'].apply(lambda x: tokenizer.tokenize(x))



# %%
# Supprimer les stop words
stop = stopwords.words('english')

df['without_stopwords']=df['token'].apply(lambda x: [word for word in x if word not in stop])



# %%
# stemmer notre list

stemmer = PorterStemmer()
df['PorterStemmer'] = df['without_stopwords'].apply(lambda x: [stemmer.stem(word) for word in x])


# %%
df['clean'] = df['without_stopwords'].apply(lambda x: " ".join(x))


# %%
# Train test split
X = df['clean']
y = df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
X

# %%
column_text=df.columns[-1]
column_text

# %%
# C'est juste un test
vectorizertest = TfidfVectorizer(analyzer='word', strip_accents ='unicode',min_df=3 )
matrix = vectorizertest.fit_transform(X_train)
matrix.shape

# %%
matrix.getmaxprint

# %%

vectorizertest.get_feature_names_out() # C'est le dictionnaire

# %%
# Transformation of textual variables
transfo_text_TFid = Pipeline(steps=[
    ('bow', TfidfVectorizer(decode_error='ignore', analyzer='char_wb', ngram_range=(2, 2),strip_accents='unicode'))
])


# svc

# %%
# Creation of model : a ready to use pipeline for ML process

model_1      =   SVC(kernel='linear')
model_2      =   LogisticRegression(C=1e5)
model_3      =   ComplementNB()
model_4      =   MultinomialNB()

model_list = [model_1,model_2,model_3,model_4]

# %%
for model in model_list:

    model_lm = Pipeline([
        ('vectorizer', transfo_text_TFid),
        ('model', model),
    ])
    # Fit the model
    model_lm.fit(X_train, y_train)
    y_pred = model_lm.predict(X_test)
    print(model,"Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    #plt.hist(model_lm.decision_function(X_test), bins=50)
    #plt.show()



