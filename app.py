import streamlit as st
import SpamProjectV3 as sp

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


spam=sp.crea
st.title('Spam Classifier')
input =  st.text_input('Enter a message')
submit = st.button('Predict')

if submit:
    for i in spam.list_model:
        model_lm=spam.ModelCreateur(spam.X_train, spam.y_train, i)
        y_pred = spam.testModel(input,model_lm)
        if y_pred == 'spam':
            st.write(i,"it's a spam")
        else:
            st.write(i,"it's not a spam")
    """print('model utilisé:', i)
    y_pred = model_lm.predict(X_test)
    spam.AfficherScores(y_test, y_pred)"""