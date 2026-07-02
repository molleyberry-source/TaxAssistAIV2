def ask_taxassist(question):
    """
    Takes a user question, searches ChromaDB for relevant tax context,
    and feeds it into OpenAI to return an accurate, grounded answer.
    """
    # 1. Convert user question into a vector embedding array
    question_embedding = model.encode(question).tolist()

    # 2. Query the 6 closest paragraph matches out of your documents
    results = collection.query(
        query_embeddings=[question_embedding], 
        n_results=6
    )

    # 3. Safely unpack text elements into context strings
    context = ""
    if results and "documents" in results and results["documents"]:
        for doc_list in results["documents"]:
            for doc in doc_list:
                if doc and doc.strip():
                    context += doc.strip() + "\n\n"

    # Fallback safety validation
    if not context.strip():
        return "The requested information is not available in the retrieved IRAS documents."

    # 4. Clear open-book instructions matching your required output structure
    prompt = f"""You are TaxAssist AI, a professional assistant for Singapore Individual Income Tax.
You must answer the user's question using ONLY the provided IRAS context below.

OUTPUT FORMAT RULES:
- Always start your output with the prefix line: "Direct answer: " followed by the primary answer statement.
- Use crisp, clean bullet points to list individual reliefs, rules, or deductions.
- If there are additional conditions, timelines, or caveats present in the context, list them under a clearly separated "Notes:" section at the bottom.

IRAS Context:
{context}

Question:
{question}

Answer:"""

    # 5. Native Responses API call configuration matching your production platform
    try:
        response = openai_client.responses.create(
            model="gpt-4o-mini", # Highly stable production variant
            input=prompt
        )
        return response.output_text
    except Exception as e:
        return f"System Connection Notice: {str(e)}"
