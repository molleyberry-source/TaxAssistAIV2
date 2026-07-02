import os
import streamlit as st
from chatbot import ask_taxassist

st.set_page_config(
    page_title="TaxAssist AI",
    page_icon="💬",
    layout="centered"
)

# ==========================================================
# AUTOMATIC DATABASE BUILDER FOR STREAMLIT CLOUD
# If the chroma_db folder or sqlite file is missing, 
# this trigger builds it using your documents on launch.
# ==========================================================
if not os.path.exists("chroma_db") or not os.path.exists("chroma_db/chroma.sqlite3"):
    with st.spinner("Initializing IRAS Tax Knowledge Base for the first time... Please wait."):
        try:
            import data_loader
            st.success("Database compiled successfully inside Streamlit Cloud!")
        except Exception as e:
            st.error(f"Database generation failed: {str(e)}")

# ==========================================================
# Standard UI Elements
# ==========================================================
st.title("TaxAssist AI")
st.write("Singapore Individual Income Tax Assistant")
st.write("Ask a question based on the IRAS knowledge base.")

question = st.text_input("Enter your tax question:", key="user_tax_question")

if st.button("Ask TaxAssist AI"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_taxassist(question)

        st.subheader("Answer")
        st.write(answer)

st.caption("Prototype for academic research. Answers are generated using retrieved IRAS document context.")
