###################################
# Llama2 model with llama_cpp_python
###################################
# libraries
from llama_cpp import Llama

#########
# Inputs
model_path = "/model/llama-2-7b-chat.Q5_K_M.gguf"
context_size = 512
max_tokens_select = 1000
temperature_select: float=0
top_p_select: float=0.9
top_k_select: int=0

########
# Prompt
#prompt = "Can you provide a summary of Guy Debord's Societe du Spectacle?"
prompt = "What kind of pickups are on an ESP LTD Alexi Ripped?"

# prompt written in gemma prompt syntax
prompt_sytnax = "<start_of_turn>user" + prompt + "<end_of_turn>" + "<start_of_turn>model"

############
# load model
llm = Llama(
  model_path=model_path,  # The path to the model file
  n_ctx=context_size,  # The max sequence length to use - adjust based on your model's requirements
  n_threads=1,  # The number of CPU threads to use
  n_gpu_layers=-1  # Set to 0 if you want to use CPU only and -1 if you want to use all available GPUs
)

##############
# send prompt
response = llm(
    prompt,
    max_tokens=max_tokens_select, 
    temperature=temperature_select,
    top_p=top_p_select,
    top_k=top_k_select,
    echo = False
    )

##############
# get response
response_text = response['choices'][0]['text']
print(response_text)
