#!/usr/bin/env python3
import os
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import vectorstore

def repl(rag_chain: RunnableSequence) ->None:
    print("Enter 'exit' to quit.")
    while True:
        question = input(">")
        if question.lower() == 'exit':
            break
        response = rag_chain.invoke(question)
        print(f"Response: {response}")
        print("\n---\n")

def main(vectorstore_file: str, chat_file: str, model='phi-4-mini', embedding_model='nomic-embed-text') ->None:
    # check for the vectorstore file, if not found create it
    if not os.path.exists(vectorstore_file):
        print(f"Vectorstore file {vectorstore_file} not found. Creating it...")
        vectorstore.create(chat_file, embedding_model, vectorstore_file)
    embeddings = OllamaEmbeddings(model=embedding_model)
    # load the vector store from disk
    vectordb = FAISS.load_local(vectorstore_file, embeddings, allow_dangerous_deserialization=True)
    # print the number of vectors in the vector store
    print(f"Number of vectors in the vector store: {vectordb.index.ntotal}")
    # initialize local model from ollama
    llm = OllamaLLM(model=model)
    prompt = ChatPromptTemplate.from_template(
        """
        Use the following context to answer the question. 

        Context: {context}

        Question: {question}
        """
        )
    rag_chain = (
        {"context": vectordb.as_retriever(),
        "question": RunnablePassthrough(),
        } | 
        prompt |
        llm |
        StrOutputParser()
    )
    repl(rag_chain)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a RAG chain with Ollama model and FAISS vector store.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vectorstore-file", type=str, default='runchat_index', help="The path to the vectorstore file.")
    parser.add_argument("--chat-file", type=str, default='running_chats.txt', help="The path to the text file to use as context.")
    parser.add_argument("--model", type=str, default="phi-4-mini", help="The Ollama model to use.")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text", help="The Ollama Embedding Model to use.")
    args = parser.parse_args()
    main(args.vectorstore_file, args.chat_file, args.model, args.embedding_model)
