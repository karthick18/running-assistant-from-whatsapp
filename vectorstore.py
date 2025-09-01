#!/usr/bin/env python3

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def create(chat_file: str, model:str='nomic-embed-text', vectorstore_file:str='runchat_index') ->FAISS:
    # load the text file
    loader = TextLoader(chat_file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    # print the number of chunks
    print(f"Number of chunks: {len(docs)}")
    # create the embeddings
    embeddings = OllamaEmbeddings(model=model)
    # create the vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    # save the vector store to disk
    vectorstore.save_local(vectorstore_file)
    return vectorstore

def main(vectorstore_file: str, chatFile: str, model:str='nomic-embed-text') ->None:
    vectorstore: FAISS = create(chatFile, model)
    # print the number of vectors in the vector store
    print(f"Number of vectors in the vector store: {vectorstore.index.ntotal}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a vectorstore with FAISS and Ollama embeddings.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vectorstore-file", type=str, default='runchat_index', help="The path to the vectorstore file.")
    parser.add_argument("--chat-file", type=str, default='running_chats.txt', help="The path to the text file to use as context.")
    parser.add_argument("--model", type=str, default="nomic-embed-text", help="The Ollama Embedding Model to use.")
    args = parser.parse_args()
    main(args.vectorstore_file, args.chat_file, args.model)
