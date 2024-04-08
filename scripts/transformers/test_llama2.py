##################################
# transformers calling Llama2 model
##################################

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import time
import pandas as pd
from pathlib import Path

def run_llama2(llm_dir, llm_model, query, customize_setting):

    #######################################
    # 1. Load the model parameters
    #######################################
    # The quantization_config is optional; use it for very large model; it reduces memory and computational costs by representing weights and activations with lower-precision data types
    # To use quantization, uncomment the following two lines and comment out the current "model = " line
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # model = AutoModelForCausalLM.from_pretrained(llm_model,cache_dir=llm_dir, device_map="auto", quantization_config=quantization_config)
    model = AutoModelForCausalLM.from_pretrained(llm_model,cache_dir=llm_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(llm_model, cache_dir=llm_dir)

    #######################################
    # 2. Convert prompt query to tokens
    #######################################
    device = "cuda"
    model_input = tokenizer(query, return_tensors="pt").to(device)

    print(f"=== Customized setting:")
    for key, value in customize_setting.items():
        print(f"    {key}: {value}")

    #######################################
    # 3. Call model to process tokens and generate response tokens
    #######################################
    outputs = model.generate(**model_input, **customize_setting)

    #######################################
    # 4. Decode tokens to text response
    #######################################
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("====================")
    print(f"LLM model: {llm_model}")
    print(f"Query: {query}")
    print("Response: ")
    print(decoded)
    print("====================")

    #######################################
    # Logging
    #######################################
    finished_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    columns = ["llm_model", "query", "response", "finished_time"]
    row = [llm_model, query, decoded, finished_time]
    for key, value in customize_setting.items():
        columns.append(key)
        row.append(value)
    df = pd.DataFrame([row], columns=columns)
    llm_name = llm_model.split("/")[-1]
    log_file = Path(f"./log_{llm_name}.csv")
    df.to_csv(log_file, index=False, mode='a', header=not log_file.exists())


# Set up model directory info; set to your own project space if using new model 
llm_dir = "/kellogg/data/llm_models_opensource/llama2_meta_huggingface"
# Model name from Huggingface site
llm_model = "meta-llama/Llama-2-7b-chat-hf"
# llm_model = "meta-llama/Llama-2-13b-chat-hf"
# llm_model = "meta-llama/Llama-2-70b-chat-hf"

# For Llama2 chat, need to enclosed your prompt by [INST] and [/INST]
query = "[INST] Tell a fun fact about Kellogg Business School. [/INST]"

# Settings for LLM model  
customize_setting = {
    "max_new_tokens": 400,
    "do_sample": True,
    "temperature": 0.8,
}

run_llama2(llm_dir, llm_model, query, customize_setting)