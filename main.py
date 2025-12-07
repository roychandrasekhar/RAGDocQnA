import os

if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "Mozilla/5.0"
    
import sys
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

def build_vectorstore(url):
    print(f"Loading url content")
    docs = WebBaseLoader(url).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(chunks, embeddings)
    return vectordb


def build_rag_chain(url):
    vectordb = build_vectorstore(url)
    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the CONTEXT and HISTORY to answer. "
                   "If the question is greeting, casual talk, or unrelated, ignore the context and answer normally"),
        ("human",
         "HISTORY:\n{history}\n\n"
         "QUESTION: {question}\n\n"
         "CONTEXT:\n{context}\n\n"
         "Answer clearly.")
    ])

    rag_chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": RunnablePassthrough(),
            "history": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    return rag_chain

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py \"<URL>\"")
        return

    url = sys.argv[1]
    rag = build_rag_chain(url)

    chat_history = InMemoryChatMessageHistory()
    print("\n*** GROQ RAG URL specific Chatbot Ready ***")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break

        history_str = "\n".join(
            f"{msg.type.upper()}: {msg.content}"
            for msg in chat_history.messages
        )

        result = rag.invoke({
            "question": q,
            "history": history_str
        })
        print("\nAssistant:", result.content, "\n")

        chat_history.add_user_message(q)
        chat_history.add_ai_message(result.content)

if __name__ == "__main__":
    main()
