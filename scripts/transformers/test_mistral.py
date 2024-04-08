##################################
# transformers calling mistral model
##################################

from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import pandas as pd
from pathlib import Path

def run_mistral(llm_dir, llm_model, messages, customize_setting):

    #######################################
    # 1. Load the model parameters
    #######################################
    model = AutoModelForCausalLM.from_pretrained(llm_model,cache_dir=llm_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(llm_model, cache_dir=llm_dir)

    #######################################
    # 2. Convert prompt query to tokens
    #######################################
    device = "cuda"
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    print(f"=== Customized setting:")
    for key, value in customize_setting.items():
        print(f"    {key}: {value}")

    #######################################
    # 3. Call model to process tokens and generate response tokens
    #######################################
    outputs = model.generate(model_inputs, **customize_setting)

    #######################################
    # 4. Decode tokens to text response
    #######################################
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print("====================")
    print(f"LLM model: {llm_model}")
    print(f"Query: {messages[-1]['content']}")
    print("Response: ")
    print(decoded[0])
    print("====================")

    #######################################
    # Logging
    #######################################
    finished_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    columns = ["llm_model", "query", "response", "finished_time"]
    row = [llm_model, messages[-1]['content'], decoded[0], finished_time]
    for key, value in customize_setting.items():
        columns.append(key)
        row.append(value)
    df = pd.DataFrame([row], columns=columns)
    llm_name = llm_model.split("/")[-1]
    log_file = Path(f"./log_{llm_name}.csv")
    df.to_csv(log_file, index=False, mode='a', header=not log_file.exists())


# Set up model directory info; set to your own project space if using new model 
llm_dir = "/kellogg/data/llm_models_opensource/mistral_mistralAI"
# Model name from Huggingface site
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"

# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

messages = [
    {"role": "user", "content": "Tell a fun fact about Kellogg Bussiness School."},
]

# Settings for LLM model  
customize_setting = {
    "max_new_tokens": 400,
    "do_sample": True,
    "temperature": 0.8,
}

run_mistral(llm_dir, llm_model, messages, customize_setting)