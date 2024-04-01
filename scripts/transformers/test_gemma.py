from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import pandas as pd
from pathlib import Path

start_time = time.time()

# Set up model directory info; set to your own project space if using new model 
llm_dir = "/kellogg/data/llm_models_opensource/gemma_google"
# Model name from Huggingface site
llm_model = "google/gemma-7b-it"
model = AutoModelForCausalLM.from_pretrained(llm_model,cache_dir=llm_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(llm_model, cache_dir=llm_dir)

print(f"=== Loading time: {time.time() - start_time} seconds")

query = "Tell a fun fact about Kellogg Business School."

device = "cuda"
model_input = tokenizer(query, return_tensors="pt").to(device)

customize_setting = {
    "max_new_tokens": 400,
    "do_sample": True,
    "temperature": 0.8,
}
print(f"=== Customized setting:")
for key, value in customize_setting.items():
    print(f"    {key}: {value}")

outputs = model.generate(**model_input, **customize_setting)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("====================")
print(f"LLM model: {llm_model}")
print(f"Query: {query}")
print("Response: ")
print(decoded)
print("====================")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
finished_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print(f"Finished at: {finished_time}")
print("====================")

# Logging
columns = ["llm_model", "query", "response", "finished_time"]
row = [llm_model, query, decoded, finished_time]
for key, value in customize_setting.items():
    columns.append(key)
    row.append(value)
df = pd.DataFrame([row], columns=columns)
llm_name = llm_model.split("/")[-1]
log_file = Path(f"./log_{llm_name}.csv")
df.to_csv(log_file, index=False, mode='a', header=not log_file.exists())