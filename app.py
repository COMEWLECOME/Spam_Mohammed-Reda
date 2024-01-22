import streamlit as st
import SpamProjectV3 as spam


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