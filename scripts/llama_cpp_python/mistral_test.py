#################################
# Mistral test code with llama_cpp
#################################

# libraries
from llama_cpp import Llama

# inputs
my_model_path = "/model/mistral-7b-v0.1.Q5_K_S.gguf"
CONTEXT_SIZE = 512

# load the model
llm = Llama(
  model_path=my_model_path,  # The path to the model file
  n_ctx=CONTEXT_SIZE,  # The max sequence length to use - adjust based on your model's requirements
  n_threads=10,  # The number of CPU threads to use
  n_gpu_layers=0  # Set to 0 if you want to use CPU only and -1 if you want to use all available GPUs
)

# send prompts
#prompt = """
#[INST] What's the most popular ice cream flavor? Answer with one or two words. For example: 'mint chocolate chip'
#[/INST]
#"""

prompt = """<s>[INST] You are a helpful assistant. What's the frequency, Kenneth? Please answer by providing a
randomly generated frequency. DO NOT quote R.E.M.  Do not tell me what you think about Generation Alpha only to be
proven wrong. Do not respond to this question more than once.  Here is an example answer:
[/INST]'20 Hz'</s>
[INST] Plesae provide only one frequency.[/INST]
"""

temp: float = 0
response = llm(prompt, max_tokens=1000, temperature=temp)
response_text = response['choices'][0]['text']
print(response_text)
