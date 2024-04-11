######################################
# Gemma Workflow with llama_cpp_python
######################################
# libraries
from llama_cpp import Llama
import pandas as pd
import os
import time

#########
# Inputs
#########

# model
model_path = "/kellogg/software/llama_cpp/models/gemma-7b-it.gguf"

# prompts
prompts = ["What is the main idea of Guy Debord's Societe du Spectacle?", 
           "What kind of pickups are on an ESP LTD Alexi Ripped?", 
           "How does Allama Iqbal's concept of the khudi relate to Nietzsche's Ubermensch?"
]

# output
output_file = "/kellogg/software/llama_cpp/output/gemma_test.csv"

# settings
context_size = 512 # The max sequence length to use - adjust based on your model's requirements
threads = 1 # The number of CPU threads to use
gpu_layers = -1 # Set to 0 if you want to use CPU only and -1 if you want to use all available GPUs
max_tokens_select = 1000
temperature_select: float=0 
top_p_select: float=0.9
top_k_select: int=0
include_prompt = False

############
# Functions
############

# get prompt response
def get_completion(llm, prompt, max_tokens_select, temperature_select, top_p_select, top_k_select, include_prompt):
  try:

    # send prompt
    response = llm(
      prompt,
      max_tokens=max_tokens_select, 
      temperature=temperature_select,
      top_p=top_p_select,
      top_k=top_k_select,
      echo = include_prompt)
    
    # get response
    response_text = response['choices'][0]['text']
    return response_text
  
  except Exception as e:
    print(f"An error occurred: {str(e)}")
    return None

# save results to a df
def save_results(prompt, response, run_time):

    # create empty df
    results_df = pd.DataFrame(columns=['prompt', 'response', 'run_time'])
    
    # create df from current row
    row_df = pd.DataFrame({
        'prompt': [prompt],
        'response': [response],
        'run_time': [run_time]
    })
    
    # combine
    results_df = pd.concat([results_df, row_df], ignore_index=True)
    
    # return dataframe
    return results_df

######
# RUN
######

def main():
   llm = Llama(model_path=model_path,  
               n_ctx=context_size,  
               n_threads=threads,  
               n_gpu_layers=gpu_layers)

   for p in prompts:
        # run
        start_time = time.time()
        response = get_completion(llm, p, max_tokens_select, temperature_select, top_p_select, top_k_select, include_prompt)
        run_time = time.time() - start_time

        # print results
        print("========================")
        print(f"Prompt: {p}")
        print(f"Response: {response}")
        print(f"Run Time: {run_time}")
        print("========================")

        # save progress
        results_df = save_results(p, response, run_time)
        results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()