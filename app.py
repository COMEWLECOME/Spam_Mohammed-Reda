import streamlit as st
import SpamProjectV3 as spam


st.title('Spam Classifier')
input =  [st.text_input('Enter a message')]
submit = st.button('Predict')

if submit:
    for i in spam.list_model:
        model_lm=spam.ModelCreateur(spam.X_train, spam.y_train, i)
        st.write(i,':',spam.testModel(input,model_lm))
    """print('model utilisé:', i)
    y_pred = model_lm.predict(X_test)
    spam.AfficherScores(y_test, y_pred)"""