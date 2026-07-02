import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(
    name="iras_tax_knowledge"
)


def ask_taxassist(question):
    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=6
    )

    context = ""

    if results and "documents" in results and results["documents"]:
        for doc in results["documents"][0]:
            if doc:
                context += doc + "\n\n"

    prompt = f"""
You are TaxAssist AI, a professional assistant for Singapore Individual Income Tax.

Use ONLY the IRAS context below to answer the question.

Rules:
- Give the direct answer first.
- Do not guess.
- Do not use outside knowledge.
- Do not include unrelated tax topics.
- If the answer is not found in the context, say:
  "The information is not available in the retrieved IRAS documents."

Question:
{question}

IRAS Context:
{context}

Answer:
"""

    try:
        response = openai_client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )

        return response.output_text

    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"