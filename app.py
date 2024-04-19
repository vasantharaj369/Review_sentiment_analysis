import streamlit as st
import pickle
import regex as re
import base64
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
port = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('img.png')


def porter(x):
    x = x.lower()
    x = re.sub('[^a-zA-Z]', ' ', x)
    x = x.split()
    y = []
    for i in x:
        if i not in stopwords.words('english'):
            y.append(port.stem(i))
    review =' '.join(y)

    return review

col1, col2, col3 = st.columns([2, 4, 1])
with col1:
    st.write('')
with col2:
    st.header('Review Classifier')
    st.write('')
with col3:
    st.write('')


value = st.text_area('Enter your review', height= 150)
st.write('')

if st.button('Predict'):

    clean = porter(value)
    X = cv.transform([clean]).toarray()
    result = model.predict(X)
    if result == 0:
        st.header('Positive')
    else:
        st.header('Negative')