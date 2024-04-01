# Documentation
This directory contains four scripts for running retrieval augmented generation.  

The "e5" scripts uses open source "intfloat/multilingual-e5-large-instruct" embedding model:  
proc_llamaindex_embed_e5_full.py   
proc_llamaindex_embed_e5_short.py   

The "openAI" scripts use "text-embedding-3-small" embedding model from openAI and expects openAI API key in .env file:  
proc_llamaindex_embed_openAI_full.py   
proc_llamaindex_embed_openAI_short.py    

The "short" script is simple and straightforword.   
The "full" script saves embedding vector store and loads vector store if exists. It also makes use of a more complicated and customizable tree-summarize response mode.