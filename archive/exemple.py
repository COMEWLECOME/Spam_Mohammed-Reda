import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



url = "SMSSpamCollection.txt"
df = pd.read_csv(url, sep='\t', header=None )
df.rename(columns={0:'label',1:'mail'}, inplace=True)
df=df.head(20)
X = df['mail']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train

vectorizer = CountVectorizer()

classifier = MultinomialNB()

pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

