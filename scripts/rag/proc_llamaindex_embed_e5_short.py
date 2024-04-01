# === Load libraries
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

start_time = time.time()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# === Download web content
url_link = "https://mistral.ai/news/announcing-mistral-7b/"
response = requests.get(url_link)
soup = BeautifulSoup(response.content, 'html.parser')
webpage_content = soup.get_text().strip()
data_folder = Path("./test_data")
if not data_folder.exists():
    data_folder.mkdir()
txt_file = data_folder / "webpage_content.txt"
with open(txt_file, "w") as f:
    f.write(webpage_content)

# Settings for embedding model
embedding_name = "intfloat/multilingual-e5-large-instruct"
embedding_chunk_size = 512
embed_model = HuggingFaceEmbedding(model_name=embedding_name, max_length=512)

# Settings for LLM
LLAMA2_13B_CHAT = "/kellogg/data/llm_models_opensource/llama2_meta_huggingface/models--meta-llama--Llama-2-13b-chat-hf/snapshots/29655417e51232f4f2b9b5d3e1418e5a9b04e80e"
selected_model = LLAMA2_13B_CHAT
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # generate_kwargs={"temperature": 0.5, "top_p": 0.9, "top_k": 2, "do_sample": True},
)

# === 1. Load
documents = SimpleDirectoryReader(data_folder).load_data()

# Set up FAISS vector store
d = 1024 # embedding dimension
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, 
    # === 2. Transform
    transformations=[SentenceSplitter(chunk_size=embedding_chunk_size)],
    # === 3. Embed
    embed_model=embed_model,
    # === 4. Store
    storage_context=storage_context
)

# === 5. Retrieve
query_engine = index.as_query_engine(llm=llm)

print("====================================")
print(f"Selected LLM: {selected_model}")
query = "[INST]What is a Mistral 7B language model?[/INST]"
response = query_engine.query(query)
print("====================================")
print(f"Query: {query}")
print("Response: ")
print(response)
print("====================================")

print(f"Execution time: {time.time() - start_time} seconds")