###################################################
# Toy Marketing Study - Does this ad appeal to you?
###################################################
# libraries
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import pandas as pd
import os
import time

#########
# Inputs
#########

llm_model = "llava-hf/bakLlava-v1-hf"
llm_dir = "/kellogg/data/llm_models_opensource/bakLlava"

prompt = """
USER: <image>\nYou are thirsty young adult between age 25 and 30 taking a marketing survey.
Can you describe if this soft drink ad appeals to you?
\nASSISTANT:
"""

output_file = "/kellogg/software/llama_cpp/output/ad_results.csv"
ad_dir = "/kellogg/software/llama_cpp/code/ads"


############
# Functions
############

# load model and processor
def load_model(llm_model, llm_dir):
    model = LlavaForConditionalGeneration.from_pretrained(
        llm_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=llm_dir,
    ).to(0)

    processor = AutoProcessor.from_pretrained(llm_model)

    return model, processor

# run bakllava
def run_bakllava(model, processor, image_file, prompt):
    # open image
    raw_image = Image.open(image_file)

    # process image
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    # generate response
    output = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    output = processor.decode(output[0][2:], skip_special_tokens=True)
    return output


# save results to a df
def save_results(prompt, image_file, response, run_time):

    # create empty df
    results_df = pd.DataFrame(columns=['prompt', 'image_file', 'response', 'run_time'])

    # create df from current row
    row_df = pd.DataFrame({
        'prompt': [prompt],
        'image_file': [image_file],
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
   # list of files from a directory
    ads = [os.path.join(ad_dir, f) for f in os.listdir(ad_dir) if os.path.isfile(os.path.join(ad_dir, f))]

    # load model
    model, processor = load_model(llm_model, llm_dir)

    # loop over
    for ad in ads:
        # run
        start_time = time.time()
        response = run_bakllava(model, processor, ad, prompt)
        run_time = time.time() - start_time

        # print results
        print("========================")
        print(f"Ad: {ad}")
        print(f"Response: {response}")
        print(f"Run Time: {run_time}")
        print("========================")

        # save progress
        results_df = save_results(prompt, ad, response, run_time)
        results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
