from pathlib import Path
import os
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import ChatPromptTemplate
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from transformers import BitsAndBytesConfig
import requests
from bs4 import BeautifulSoup


def prep_text(workdir, url_link):
    response = requests.get(url_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    webpage_content = soup.get_text().strip()

    data_path = workdir / "LlamaIndex_data"
    if not data_path.exists():
        data_path.mkdir()

    txt_file = data_path / "webpage_content.txt"
    with open(txt_file, "w") as f:
        f.write(webpage_content)

def process_llamaindex(workdir):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    api_file = ".env"
    with open(api_file, 'r') as f:
        os.environ['OPENAI_API_KEY'] = f.read().strip()

    # Settings for embedding model
    embedding_model = "text-embedding-3-small"
    embedding_chunk_size = 512
    embed_model = OpenAIEmbedding(model=embedding_model)

    # Settings for LLM
    LLAMA2_7B_CHAT = "/kellogg/data/llm_models_opensource/llama2_meta_huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"
    # LLAMA2_13B_CHAT = "/kellogg/data/llm_models_opensource/llama2_meta_huggingface/models--meta-llama--Llama-2-13b-chat-hf/snapshots/29655417e51232f4f2b9b5d3e1418e5a9b04e80e"
    selected_model = LLAMA2_7B_CHAT
    # selected_model = LLAMA2_13B_CHAT
    # Optional quantization for loading large models
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=512,
        tokenizer_name=selected_model,
        model_name=selected_model,
        device_map="auto",
        # model_kwargs={"quantization_config": quantization_config},
    )

    PERSIST_DIR = workdir / "LlamaIndex_storage_faiss"
    data_path = workdir / "LlamaIndex_data"

    if not PERSIST_DIR.exists():
        print("== Processing data ... ")
        # load the documents and create the index
        documents = SimpleDirectoryReader(data_path).load_data()

        # FAISS vector store
        d = 1536 # embedding dimension
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, 
            transformations=[SentenceSplitter(chunk_size=embedding_chunk_size)],
            embed_model=embed_model, 
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

    # Query with Tree summarizer
    system_prompt = ChatMessage(
        content=(
        f"[INST]<<SYS>>\nYou are a helpful assistant.<</SYS>>\n" 
        ),
        role=MessageRole.SYSTEM,
    )
    user_prompt = ChatMessage(
        content=(
        "Context information from multiple sources is below.\n---------------------\n{context_str}\n---------------------\nGiven the information from multiple sources and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: [/INST]"
        ),
        role=MessageRole.USER,
    )

    chat_prompt = ChatPromptTemplate(message_templates=[system_prompt, user_prompt])
    summarizer = TreeSummarize(verbose=True, summary_template=chat_prompt)

    query_engine = index.as_query_engine(
        llm = llm,
        response_synthesizer=summarizer,
    )

    print("====================================")
    print(f"Selected LLM: {selected_model}")
    print("====================================")
    query = "What is a Mistral 7B language model?"
    response = query_engine.query(query)
    print("====================================")
    print(f"Query: {query}")
    print("Response: ")
    print(response)
    print("====================================")


# Replace your_file_directory with your project directory string
file_dir = your_file_directory
workdir = Path(file_dir)
url_link = "https://mistral.ai/news/announcing-mistral-7b/"
prep_text(workdir, url_link)

process_llamaindex(workdir)
