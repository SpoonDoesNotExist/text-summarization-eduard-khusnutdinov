import streamlit as st
from transformers import pipeline
import time


@st.cache_resource
def load_summarizer():
    return pipeline("summarization")


summarizer = load_summarizer()

st.title("Text Summarization App")
st.write("Enter text in the field below to generate a summary.")

input_text = st.text_area("Input Text", height=200)

if st.button("Generate Summary"):
    if input_text.strip():
        with st.spinner("Generating summary, please wait..."):
            time.sleep(1)

            summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']  # noqa

            st.success("Summary generated successfully!")
            st.write("### Summary")
            st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
