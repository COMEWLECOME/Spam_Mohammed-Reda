import streamlit as st
import pandas as pd 

url = "SMSSpamCollection.txt"
df = pd.read_csv(url, sep='\t', header=None )

df.rename(columns={0:'type',1:'mail'}, inplace=True)

# st.sidebar → colonne de gauche
st.sidebar.title('Configuration')
nl = st.sidebar.slider('Lignes',
min_value=0,
max_value=min(50,df.shape[0]))
# partie centrale
st.table(df.iloc[0:nl])