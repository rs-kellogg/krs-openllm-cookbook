#################################
# Llama2 test code with llama_cpp
#################################

# libraries
from llama_cpp import Llama

# inputs
my_model_path = "/model/llama-2-7b-chat.Q5_K_M.gguf"
max_l = 5000

# load the model
llm = Llama(
  model_path=my_model_path,  # The path to the model file
  n_threads=4,  # The number of CPU threads to use
  n_gpu_layers=0  # Set to 0 if you want to use CPU only and -1 if you want to use all available GPUs
)

message = "What's the most popular ice cream flavor?"
response = llm(message, max_tokens=max_l, temperature=0)
generated_text = response["choices"][0]["text"]
print(f"System: {generated_text}")

