# ==========================================================
# TaxAssist AI - Chatbot Logic (Production Version)
# Purpose:
#   1. Capture user income tax query.
#   2. Convert query to a vector embedding using all-MiniLM-L6-v2.
#   3. Retrieve the top 6 overlapping paragraph chunks from ChromaDB.
#   4. Deliver grounded text context to the OpenAI Responses API.
# ==========================================================

import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables (Local .env file)
load_dotenv()

# Initialize official OpenAI client
# Reads OPENAI_API_KEY from your local .env or Streamlit Secrets online automatically
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the exact same embedding model used in your data loader script
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to your local ChromaDB vector directory
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Access your custom IRAS tax collection
collection = chroma_client.get_or_create_collection(name="iras_tax_knowledge")


def ask_taxassist(question):
    """
    Takes a user question, searches ChromaDB for relevant tax context,
    and feeds it into OpenAI to return an accurate, grounded answer.
    """
    # 1. Convert user question into a vector embedding array
    question_embedding = model.encode(question).tolist()

    # 2. Optimized: Query the 6 closest paragraph matches out of your documents
    results = collection.query(
        query_embeddings=[question_embedding], 
        n_results=6
    )

    # 3. Safely unpack and combine retrieved text chunks into a single context string
    context = ""
    if results and "documents" in results and results["documents"]:
        # ChromaDB nests documents inside an outer list; loop through safely
        for doc_list in results["documents"]:
            for doc in doc_list:
                if doc and doc.strip():
                    context += doc.strip() + "\n\n"

    # 4. Create your strict open-book prompt with explicit formatting guidelines
    prompt = f"""
You are TaxAssist AI, a professional assistant for Singapore Individual Income Tax.

Answer the user's question using ONLY the IRAS context provided below.
If the answer is not found in the context, say that the information is not available in the retrieved IRAS documents.

OUTPUT FORMAT RULES:
- Always start your output with the prefix line: "Direct answer: " followed by the primary answer statement.
- Use crisp, clean bullet points to list individual reliefs, rules, or deductions.
- If there are additional conditions, timelines, or caveats present in the context, list them under a clearly separated "Notes:" section at the bottom.
- Include only information that is directly relevant to the user's question. Do not include unrelated tax reliefs, deductions, rebates, or other topics.

Question:
{question}

IRAS Context:
{context}

Answer:
"""

    # 5. Send the structured context and input to the OpenAI infrastructure
    try:
        response = openai_client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"


# ==========================================================
# Interactive Terminal Chat Mode
# Run 'python chatbot.py' to start chatting live
# ==========================================================
if __name__ == "__main__":
    print("\n=========================================================")
    print("🇸🇬 Welcome to TaxAssist AI (Interactive Terminal Mode) 🇸🇬")
    print("Type your tax question below. Type 'exit' or 'quit' to stop.")
    print("=========================================================\n")
    
    while True:
        # Prompting for user input dynamically via the console
        user_input = input("Ask your tax question (or type 'exit'): ")
        
        # Safe exit routine
        if user_input.strip().lower() in ["exit", "quit"]:
            print("\nThank you for using TaxAssist AI. Keep track of your deadlines! Goodbye!")
            break
            
        # Bypass empty string submissions
        if not user_input.strip():
            continue
            
        print("\nThinking...")
        
        # Execute the RAG lookup and OpenAI inference pipelines
        answer = ask_taxassist(user_input)
        
        print("\n=== ANSWER ===")
        print(answer)
        print("=========================================================\n")
