import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from utils import clean_text
from utils import calculate_similarity


def create_streamlit_app(llm,  clean_text):
    st.title("📧 Cold Mail or Cover Letter Generator")
    st.subheader("Please enter job decription", divider=True)
    text_input = st.text_area("")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            data = clean_text(text_input)
            job = llm.extract_jobs(data)
            email = llm.write_mail(job)
            if data:
               accuracy = calculate_similarity(data, email)
               print(accuracy)
            st.code(email, language='markdown')
            st.code(accuracy, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain,  clean_text)


