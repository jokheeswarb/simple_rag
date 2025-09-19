# streamlit_app.py
import os
import streamlit as st
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import openai


class SimpleRAG:
    def __init__(self):
        self.vectorizer = None
        self.index = None
        self.docs = []

    def add_documents(self, docs):
        self.docs.extend(docs)
        self.vectorizer = TfidfVectorizer()
        doc_vectors = self.vectorizer.fit_transform(self.docs).toarray().astype("float32")
        doc_vectors = self._normalize(doc_vectors)
        dim = doc_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(doc_vectors)

    def _normalize(self, vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-9)

    def search(self, query, top_k=3):
        if not self.index:
            return []
        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        q_vec = self._normalize(q_vec)
        scores, idxs = self.index.search(q_vec, top_k)
        return [(self.docs[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i != -1]


def generate_answer(question, context_docs):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        return "Missing OPENAI_API_KEY."

    context = "\n".join([f"- {doc}" for doc in context_docs])
    prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {question}\nAnswer:"

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()


st.title("Simple RAG")
st.write("Upload text files, index them, and ask questions!")

if "rag" not in st.session_state:
    st.session_state.rag = SimpleRAG()
    st.session_state.indexed = False

# Upload text files
uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    texts = []
    for file in uploaded_files:
        content = file.read().decode("utf-8", errors="ignore")
        texts.append(content)
    st.session_state.rag.add_documents(texts)
    st.session_state.indexed = True
    st.success(f"Indexed {len(texts)} documents.")

# Ask a question
if st.session_state.indexed:
    question = st.text_input("Ask a question:")
    if question:
        hits = st.session_state.rag.search(question, top_k=3)
        passages = [doc for doc, _ in hits]

        st.subheader("Retrieved Passages")
        for i, (doc, score) in enumerate(hits, 1):
            st.markdown(f"**[{i}] (score={score:.2f})** {doc[:400]}{'...' if len(doc) > 400 else ''}")

        st.subheader("🤖 Answer")
        answer = generate_answer(question, passages)
        st.write(answer)
