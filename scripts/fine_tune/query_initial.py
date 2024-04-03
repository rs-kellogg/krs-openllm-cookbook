#############################
# Fine Tune - Initial Query
#############################

# import libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb

# model paths
llm_dir = "/kellogg/data/llm_models_opensource/mistral_mistralAI"
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# quantized model loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model loading
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, cache_dir=llm_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, cache_dir=llm_dir)

# query function
def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"

  prompt_template = """
  <s>
  [INST]
  Classify the following text into one of the four classes: 0 - World, 1 - Sports, 2 - Business, 3 - Sci/Tech.
  {query}
  [/INST]
  </s>
  <s>

  """
  prompt = prompt_template.format(query=query)
  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encodeds.to(device)
  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])

# query
query = """
Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling band 
of ultra-cynics, are seeing green again.
"""

# submit query
result = get_completion(query=query, model=model, tokenizer=tokenizer)
print(result)
