#!/usr/bin/env python3
import json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def create_by_date(chat_file: str, model:str='nomic-embed-text', vectorstore_file:str='runchat_index') ->FAISS:
    # load the text file
    # load the json file and read the daily chunk lists
    with open(chat_file, 'r', encoding='utf-8') as f:
        daily_chunks: list[dict] = json.load(f)
    # create a list of documents from the daily chunks
    text_splitter = RecursiveJsonSplitter(max_chunk_size=500)
    docs = text_splitter.create_documents(texts=daily_chunks, convert_lists=True)
    # print the number of chunks
    print(f"Number of chunks: {len(docs)}")
    # create the embeddings
    embeddings = OllamaEmbeddings(model=model)
    # create the vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    # save the vector store to disk
    vectorstore.save_local(vectorstore_file)
    return vectorstore

def create(chat_file: str, model:str='nomic-embed-text', vectorstore_file:str='runchat_index') ->FAISS:
    # load the text file
    loader = TextLoader(chat_file, encoding='utf-8', autodetect_encoding=True)
    documents = loader.load()
    if chat_file.endswith('.json'):
        text_splitter = RecursiveJsonSplitter(max_chunk_size=500)
    else:    
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

def main(by_date: bool, vectorstore_file: str, chatFile: str, model:str='nomic-embed-text') ->None:
    if by_date:
        vectorstore: FAISS = create_by_date(chatFile, model, vectorstore_file)
    else:
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
    # create by date
    parser.add_argument("--by-date", action='store_true', help="Create the vectorstore by grouping messages by date.")    
    args = parser.parse_args()
    main(args.by_date, args.vectorstore_file, args.chat_file, args.model)
