#!/usr/bin/env python3
import os
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import vectorstore

def repl(rag_chain: RunnableSequence, with_context:bool = False) ->None:
    print("Enter 'exit' to quit.")
    while True:
        question = input(">")
        if question.lower() == 'exit':
            break
        if question.strip() == '':
            continue
        response = rag_chain.invoke(question)
        if with_context:
            answer: str = response['answer']
            print(f"Response: {answer}")
            context: str = response['context']
            for doc in context:
                print(f"Source: {doc.page_content}, Metadata: {doc.metadata}")
        else:
            print(f"Response: {response}")
        print("\n---\n")

def format_docs(docs:list[Document]) ->str:
    return '\n\n'.join([doc.page_content for doc in docs])

def main(vectorstore_file: str, chat_file: str, model='phi-4-mini',
         embedding_model='nomic-embed-text', with_context:bool = False) ->None:
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
    if not with_context:
        rag_chain = (
            # restrict the number of documents to 3
            {"context": vectordb.as_retriever(search_kwargs={"k": 2}),
            "question": RunnablePassthrough(),
            } | 
            prompt |
            llm |
            StrOutputParser()
        )
    else:
        answer_rag_chain = (prompt | llm | StrOutputParser())
        rag_chain = RunnableParallel(
            {
                "context": vectordb.as_retriever(search_kwargs={"k":2}),
                "answer": ( 
                    {
                    "context": vectordb.as_retriever(search_kwargs={"k":2}),
                    "question": RunnablePassthrough(),
                    } | answer_rag_chain
                )
            }
        )
    repl(rag_chain, with_context)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a RAG chain with Ollama model and FAISS vector store.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vectorstore-file", type=str, default='runchat_index', help="The path to the vectorstore file.")
    parser.add_argument("--chat-file", type=str, default='running_chats.txt', help="The path to the text file to use as context.")
    parser.add_argument("--model", type=str, default="phi-4-mini", help="The Ollama model to use.")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text", help="The Ollama Embedding Model to use.")
    # repl with context
    parser.add_argument("--with-context", action='store_true', help="Create the answer with context.")
    args = parser.parse_args()
    main(args.vectorstore_file, args.chat_file, args.model, args.embedding_model, args.with_context)
