#python3 -m streamlit run text_streamlit.py 
import streamlit as st
import pandas as pd
url = "SMSSpamCollection.txt"
data=pd.read_csv(url, sep='\t', header=None )
data.rename(columns={0:'type',1:'mail'}, inplace=True)
data['len']=data['mail'].str.len()
data['nombre_mots']=data['mail'].str.split().str.len()
# st.sidebar → colonne de gauche
st.sidebar.title('Configuration')

n2 = st.sidebar.radio('choisis le type de mail',['spam','ham'])

nl = st.sidebar.slider('Lignes', 
    min_value=0, 
    max_value=min(50,data.shape[0]))

df=data[data['type']==n2].iloc[0:nl]


# partie centrale
st.table(df)

# Bar graph
st.bar_chart(data=df, y='len')

# Scatter plot
st.scatter_chart(data=data.iloc[:nl], x='len', y='nombre_mots')

