# Load libraries
from pathlib import Path
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import requests
from bs4 import BeautifulSoup
import time
import os

start_time = time.time()

# Save text from the url link
def prep_text(workdir, url_link):
    response = requests.get(url_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    webpage_content = soup.get_text().strip()
    webpage_content = os.linesep.join([s for s in webpage_content.splitlines() if s])

    data_path = workdir / "LlamaIndex_data"
    if not data_path.exists():
        data_path.mkdir()

    txt_file = data_path / "webpage_content.txt"
    with open(txt_file, "w") as f:
        f.write(webpage_content)

# Perform RAG using LlamaIndex
def process_llamaindex(workdir, embedding_name, embed_d, llm_model, query):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    embedding_chunk_size = 512
    embed_model = HuggingFaceEmbedding(model_name=embedding_name, max_length=512)

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=1000,
        tokenizer_name=llm_model,
        model_name=llm_model,
        device_map="auto",
    )

    PERSIST_DIR = workdir / "LlamaIndex_storage_faiss"
    data_path = workdir / "LlamaIndex_data"

    if not PERSIST_DIR.exists():
        print("== Processing data ... ")

        ##############################
        # 1. Load the documents
        ##############################
        documents = SimpleDirectoryReader(data_path).load_data()

        # Set up FAISS vector store
        faiss_index = faiss.IndexFlatL2(embed_d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, 
            ##############################
            # 2. Transform
            ##############################
            transformations=[SentenceSplitter(chunk_size=embedding_chunk_size)],
            ##############################
            # 3. Embed
            ##############################
            embed_model=embed_model, 
            ##############################
            # 4. Store
            ##############################
            storage_context=storage_context
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("== Finish saving vector store data ... ")

    else:
        print("== Storage exists. Loading data ...")

        vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR,
            vector_store=vector_store, 
        )
        index = load_index_from_storage(
            transformations=[SentenceSplitter(chunk_size=embedding_chunk_size)],
            embed_model=embed_model, 
            storage_context=storage_context,
            )
        print("== Finish loading ...")

    ##############################
    # 5. Retrieve
    ##############################
    query_engine = index.as_query_engine(llm = llm)

    print("====================================")
    print(f"Selected LLM: {llm_model}")
    print("====================================")
    response = query_engine.query(query)
    print("====================================")
    print(f"Query: {query}")
    print("Response: ")
    print(response)
    print("====================================")


# Replace your_file_directory with your project directory string
file_dir = your_file_directory
workdir = Path(file_dir)
url_link = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
prep_text(workdir, url_link)


# Settings for embedding model
# embedding_name = "intfloat/multilingual-e5-large-instruct"
embedding_name = "/kellogg/data/llm_models_opensource/e5_infloat/models--intfloat--multilingual-e5-large-instruct/snapshots/baa7be480a7de1539afce709c8f13f833a510e0a"
embed_d = 1024 # embedding dimension

# Settings for LLM
LLAMA2_7B_CHAT = "/kellogg/data/llm_models_opensource/llama2_meta_huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"
# LLAMA2_13B_CHAT = "/kellogg/data/llm_models_opensource/llama2_meta_huggingface/models--meta-llama--Llama-2-13b-chat-hf/snapshots/29655417e51232f4f2b9b5d3e1418e5a9b04e80e"
llm_model = LLAMA2_7B_CHAT
# llm_model = LLAMA2_13B_CHAT

query = "[INST]How can I use tranformers to call the Mistral 7B instruct model?[/INST]"

process_llamaindex(workdir, embedding_name, embed_d, llm_model, query)

