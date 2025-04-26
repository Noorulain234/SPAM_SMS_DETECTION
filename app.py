import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model_1=pickle.load(open('model_1.pkl','rb'))
st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("enter the message")
transformed_sms =(transform_text(input_sms))
#if st.button('predict'):

vector_input =tfidf.transform([transformed_sms])
result=model_1.predict(vector_input)[0]
if result == 1:
     st.header("spam")
else:
     st.header("not spam")



