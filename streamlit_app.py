import os
import streamlit as st
from chatbot import ask_taxassist

st.set_page_config(
    page_title="TaxAssist AI",
    page_icon="💬",
    layout="centered"
)

# Ensure database check handles paths correctly online
if not os.path.exists("chroma_db") or not os.path.exists("chroma_db/chroma.sqlite3"):
    with st.spinner("Initializing IRAS Tax Knowledge Base... Please wait."):
        try:
            import data_loader
            st.success("Knowledge Base loaded successfully!")
        except Exception as e:
            st.error(f"Initialization Note: {str(e)}")

st.title("TaxAssist AI")
st.write("Singapore Individual Income Tax Assistant")
st.write("Ask a question based on the IRAS knowledge base.")

question = st.text_input("Enter your tax question:", key="user_tax_question")

if st.button("Ask TaxAssist AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing document context..."):
            # Execute backend pipeline
            answer_output = ask_taxassist(question)
        
        # Display the result using markdown container boundaries
        st.subheader("Answer")
        st.markdown(answer_output)

st.caption("Prototype for academic research. Answers are generated using retrieved IRAS document context.")
