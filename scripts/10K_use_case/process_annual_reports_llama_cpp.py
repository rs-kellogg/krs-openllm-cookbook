###########################
#  Process Annual Reports #
###########################
# import libraries
import pandas as pd
import csv
import re
import os
import itertools

from llama_cpp import Llama


#########
# Inputs
#########

model_path = "/model/gemma-7b-it.gguf"
temperature: float=0

# Create the system prompt
system_prompt = "You are a concise assistant."

# Create the user prompt
text = " "
prompt = "Can you summarize the highlights of this text in one sentence?\n" + text

#input_dir = "/kellogg/data/EDGAR/10-K/2024"
input_dir = "/data/2024"
output_file = "/code/annual_report_output.csv"


############
# Functions
############

# clean html from 10K reports
def clean_html(html):
    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()

# obtain a list of 10K text files in a folder
def get_files_in_folder(folder_path):
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            files.append(file)
    return files

# chat response
def get_chat_response(llm, system_prompt: str, user_prompt: str, temp=0):
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        #messages = "<start of turn>user " + system_prompt + " " + user_prompt + "<end of turn> <start of turn> model"

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=1000,
            temperature=temp,
        )
        

        response_content = response['choices'][0]['message']['content'] #response['choices'][0]['text']
        return response_content

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# save new dataframe
def save_results(results_df, file_name, response):
    
    # Create a DataFrame from the current row
    row_df = pd.DataFrame({
        'file_name': [file_name],
        'response': [response]
    })
    
    # Concatenate the current row DataFrame with the existing results DataFrame
    results_df = pd.concat([results_df, row_df], ignore_index=True)
    
    # return dataframe
    return results_df

# create an empty dataframe
def create_df():
    # create an empty dataframe to save results
    results_df = pd.DataFrame(columns=['file_name', 'response'])
    return results_df


######
# Run
######

def main():

    # get list of 10K files
    files = get_files_in_folder(input_dir)

    # create an empty df to save results
    results_df = create_df()

    # load the model
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048)
    
    for file in files[0:5]:
        print(file)
        file_name = file

        # load 10K file
        with open(os.path.join(input_dir, file), 'r') as f:
            ten_k_text = f.read()
        
        # clean html
        ten_k_text = clean_html(ten_k_text)

        
        # obtain the second occurrence of "Discussion and Analysis of Financial Condition" with wildcards
        mda_matches = re.finditer(r"Discussion[\s,.-]*and[\s,.-]*Analysis[\s,.-]*of[\s,.-]*Financial[\s,.-]*Condition", ten_k_text, re.IGNORECASE)
        second_match = next(itertools.islice(mda_matches, 1, None), None)  # get the second match

        if second_match is not None:
            mda_start = second_match.start()
            mda_end = second_match.end()
            mda_text = ten_k_text[mda_end:]
            mda_text = mda_text[:1000]  # limit to the first 1000 words
            print(mda_text)
        else:
            mda_text = "nothing found"
            print(mda_text)

        # user prompt
        user_prompt = prompt + " " + mda_text
        print(user_prompt)

        # send prompt
        response = get_chat_response(llm, system_prompt, user_prompt)
        print(response)


        # save results
        results_df = save_results(results_df, file_name, response)
        output_path = output_file
        results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()


