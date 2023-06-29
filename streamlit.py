import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sem.logic import Tokens
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
from PIL import Image
import pytesseract


def main():
    def extract_text(image):
        image = Image.open(image)
        text = pytesseract.image_to_string(image)
        return text

    def cosine(r_list):
        cv = CountVectorizer()
        matrix = cv.fit_transform([' '.join(lst) for lst in r_list])
        cos_sim = cosine_similarity(matrix)
        similarity_rate = cos_sim[0][1] * 100
        return (f"Similarity Rate: {similarity_rate:.2f}%")


    def preprocessing(text):

        text_p = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text_p)
        stop_words = set(stopwords.words('english'))
        without_stop_words = []
        for words in tokens:
            if words not in stop_words:
                without_stop_words.append(words)
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(words) for words in without_stop_words]
        final_text = [x for i, x in enumerate(lemmatized_words) if x not in lemmatized_words[:i]]
        return final_text

    st.title("HTML Input Example")

    col1, col2 = st.columns(2)


    # Column 1: Text input
    with col1:
        st.markdown('<style>div.file_input.UploadFile > label{height: 200px !important;}</style>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="png")

    # Column 2: PDF file uploader
    with col2:
        st.markdown('<style>div.Widget.row-widget.stRadio > label{height: 200px !important;}</style>',
                    unsafe_allow_html=True)
        user_input = st.text_input("Enter your name", "John Doe", key='text_input')

    if st.button("Submit"):
        temp = []
        # Call the function when the button is clicked
        uncleaned_text =extract_text(uploaded_file)
        cleaned_text_resume=preprocessing(uncleaned_text)
        cleaned_text_jd = preprocessing(user_input)
        temp.append(cleaned_text_resume)
        temp.append(cleaned_text_jd)
        percentage = cosine(temp)
        st.write(percentage)



if __name__ == "__main__":
    main()
