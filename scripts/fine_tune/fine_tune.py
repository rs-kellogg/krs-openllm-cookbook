#######################################
# Fine Tuning using PEFT, LoRA, and SFT
#######################################

# import libraries
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
import bitsandbytes as bnb
from lora import LoraConfig, get_peft_model, PeftModel
from sft import SFTTrainer
from datasets import load_from_disk

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

# load the train and test datasets
train_data = load_from_disk("/kellogg/data/llm_models_opensource/mistral_mistralAI/fine_tune/ag_news/data/train_data.jsonl")
test_data = load_from_disk("/kellogg/data/llm_models_opensource/mistral_mistralAI/fine_tune/ag_news/data/test_data.jsonl")

# find modules in the model
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
modules = find_all_linear_names(model)

# configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# calculate the number of trainable parameters
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

# set padding token
tokenizer.pad_token = tokenizer.eos_token

# clear GPU memory cache
torch.cuda.empty_cache()

# setup and start training
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="text",
    peft_config=lora_config,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

# save the model
new_model = "mistralai-Code-Instruct-ag_news" #Name of the model you will be pushing to huggingface model hub
trainer.model.save_pretrained(new_model)

# merge the models
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# save the merged model
merged_model.save_pretrained(new_model, safe_serialization=True)
tokenizer.save_pretrained(new_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
