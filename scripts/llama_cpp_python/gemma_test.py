#############################
# Llama_cpp using Gemma model
#############################
# libraries
from llama_cpp import Llama

# Inputs
model_path ="/model/gemma-7b-it.gguf"
CONTEXT_SIZE = 512
temperature: float=0

# basic prompt
#prompt = "Can you provide a summary of Guy Debord's Societe du Spectacle?"

# prompt written in gemma prompt syntax
prompt = """
<start_of_turn>user
Can you provide a summary of Guy Debord's Societe du Spectacle?"<end_of_turn>
<start_of_turn>model
"""

# LOAD THE MODEL
llm = Llama(
  model_path=model_path,  # The path to the model file
  n_ctx=CONTEXT_SIZE,  # The max sequence length to use - adjust based on your model's requirements
  n_threads=1,  # The number of CPU threads to use
  n_gpu_layers=-1  # Set to 0 if you want to use CPU only and -1 if you want to use all available GPUs
)

# send prompts
response = llm("Can you provide a concise summary of Debord's Societe du Spectacle?", max_tokens=1000, 
temperature=temperature)
response_text = response['choices'][0]['text']
print(response_text)
