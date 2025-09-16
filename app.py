import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache FAISS vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def main():
    st.title("Ask Chatbot (Groq Free Model)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input("Pass your prompt here")
    if not prompt:
        return

    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you don’t know the answer, just say you don’t know. Don’t make up anything. 
    Only answer from the given context.

    Context: {context}
    Question: {question}

    Answer:
    """

    try:
        vectorstore = get_vectorstore()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # Free Groq-hosted model
                temperature=0.0,
                groq_api_key=os.environ["GROQ_API_KEY"],  # Add this in Streamlit secrets
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        # Use 'question' key
        response = qa_chain.invoke({'question': prompt})

        result = response["result"]
        source_info = "\n".join(
            [f"- {doc.metadata.get('source','Unknown')} | {doc.page_content[:200]}..."
             for doc in response["source_documents"]]
        )

        result_to_show = f"{result}\n\n**Source Docs:**\n{source_info}"
        st.chat_message('assistant').markdown(result_to_show)
        st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
