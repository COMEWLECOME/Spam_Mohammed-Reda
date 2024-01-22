import pandas as pd

df = pd.read_csv("SMSSpamCollection.txt", sep='\t', header=None )
df.rename(columns={0:'type',1:'mail'}, inplace=True)
    
spam_messages = df[df['type'] == 'spam']['mail']
ham_messages = df[df['type'] == 'ham']['mail']


spam_words = set(' '.join(spam_messages).split())
ham_words = set(' '.join(ham_messages).split())

spam_only_words = spam_words - ham_words

print(spam_only_words)

with open('spam_words1.txt', 'w') as f:
    for word in spam_only_words:
        f.write(word + '\n')