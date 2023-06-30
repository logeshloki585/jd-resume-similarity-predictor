import streamlit as st
from PIL import Image
import pytesseract
import openai
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    def extract_text(image):
        image = Image.open(image)
        text = pytesseract.image_to_string(image)
        return text

    def extract_skills_experience(resume,jd,key):
        text = "resume :" + resume + "job description :"+ jd
        openai.api_key = key
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




    st.header("RESUME - JOB DESCRIPTION ANALYZER")

    user_API_KEY = st.text_input("ENTER OPEN-API-KEY", "", key='key_input')

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

        uncleaned_text =extract_text(uploaded_file)
        percentage = extract_skills_experience(uncleaned_text,user_input_jd,user_API_KEY)
        st.write(percentage)



if __name__ == "__main__":
    main()
