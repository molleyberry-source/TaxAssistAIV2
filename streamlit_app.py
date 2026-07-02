import streamlit as st
from chatbot import ask_taxassist

# 1. Page Configuration
st.set_page_config(
    page_title="TaxAssist AI",
    page_icon="💬",
    layout="centered"
)

st.title("TaxAssist AI")
st.write("Singapore Individual Income Tax Assistant")
st.write("Ask a question based on the IRAS knowledge base.")

# 2. Initialize Chat History Session Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your TaxAssist AI. Ask me anything about Singapore individual income tax."}
    ]

# 3. Render Existing Chat Bubble History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle Live Dynamic Chat Input
if user_query := st.chat_input("Enter your tax question here..."):
    # Show user message instantly
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Compute and stream AI response from your RAG chatbot script
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_response = ask_taxassist(user_query)
            st.markdown(ai_response)
            
    # Save output to session state memory
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# 5. Fixed Research Caption Footer
st.markdown("---")
st.caption("Prototype for academic research. Answers are generated using retrieved IRAS document context.")
