import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from utils import clean_text


def create_streamlit_app(llm,  clean_text):
    st.title("📧 Cold Mail or Cover Letter Generator")
    st.subheader("Please enter job decription", divider=True)
    text_input = st.text_area("")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            print(text_input)
            data = clean_text(text_input)
            job = llm.extract_jobs(data)
            email = llm.write_mail(job)
            st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain,  clean_text)


