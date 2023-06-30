import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sem.logic import Tokens
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
from PIL import Image
import pytesseract
import requests
import re
import json
import openai
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    def extract_text(image):
        image = Image.open(image)
        text = pytesseract.image_to_string(image)
        return text

    def extract_skills_experience(resume,jd):
        text = "resume :" + resume + "job description :"+ jd
        openai.api_key = os.getenv('GPT_API_KEY')
        
        print(openai.api_key)
        prompt = "give me only the percentage of similarity for this resume and job descrption:\n\n" + text + "\n\n---\n\nInput:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.3,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()

    def get_entity(text):
        resume_data = []
        include = 'extract named entity'
        # YOUR_TOKEN = "f69b378878d141f29f40aaaf093e5600"
        url = f'https://api.dandelion.eu/datatxt/nex/v1/?include={include}&text={text}&token=f69b378878d141f29f40aaaf093e5600'
        # print(url)
        response = requests.get(url)
        data = json.loads(response.text)
        for i in data['annotations']:
            resume_data.append(i['spot'])
        return (resume_data)

    def cosine(r_list):
        cv = CountVectorizer()
        matrix = cv.fit_transform([' '.join(lst) for lst in r_list])
        cos_sim = cosine_similarity(matrix)
        similarity_rate = cos_sim[0][1] * 100
        return (f"Similarity Rate: {similarity_rate:.2f}%")


    def preprocessing(text):
        data = ''
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
        for i in final_text:
            data = data + ' ' + i
        return data

    st.header("RESUME - JOB DESCRIPTION ANALYZER")

    col1, col2 = st.columns(2)


    # Column 1: Text input
    with col1:
        st.markdown('<style>div.file_input.UploadFile > label{height: 200px !important;}</style>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader("UPLOAD RESUME", type=None)

    # Column 2: PDF file uploader
    with col2:
        st.markdown('<style>div.Widget.row-widget.stRadio > label{height: 200px !important;}</style>',
                    unsafe_allow_html=True)
        user_input_jd = st.text_input("ENTER JOB-DESCRIPTION", "", key='text_input')

    if st.button("Submit"):
        temp = []
        # Call the function when the button is clicked
        uncleaned_text =extract_text(uploaded_file)

        # cleaned_text_resume=preprocessing(uncleaned_text)
        # cleaned_text_jd = preprocessing(user_input)
        #
        # entity_of_resume = get_entity(cleaned_text_resume)
        # entity_of_jd = get_entity(cleaned_text_jd)
        #
        # temp.append(entity_of_resume)
        # temp.append(entity_of_jd)
        # percentage = cosine(temp)
        percentage = extract_skills_experience(uncleaned_text,user_input_jd)
        st.write(percentage)



if __name__ == "__main__":
    main()
