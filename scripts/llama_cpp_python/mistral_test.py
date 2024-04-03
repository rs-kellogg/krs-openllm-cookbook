###################################
# Mistral model with llama_cpp_python
###################################
# libraries
from llama_cpp import Llama

#########
# Inputs
model_path = "/model/mistral-7b-v0.1.Q5_K_S.gguf"
context_size = 512
max_tokens_select = 20
temperature_select: float=0
top_p_select: float=0.9
top_k_select: int=0

########
# Prompt
prompt = "Please randomly select an ice cream flavor.  Here is an example: rocky road. Now you give me one."
prompt_syntax = """<s>[INST] Please randomly select an ice cream flavor.  Here is an example answer:
[/INST]'rocky road'</s>
"""

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
