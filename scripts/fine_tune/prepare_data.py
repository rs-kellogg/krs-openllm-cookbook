#############################
# Fine Tune - Prepare Dataset
#############################

# import libraries
import json
from datasets import load_dataset

# create a prompt/response row
def create_text_row(row):
    text_row = f"<s>[INST] Classify the following text into one of the four classes: 0 - World, 1 - Sports, 2 - Business, 3 - Sci/Tech. {row['text']} [/INST] \\n {row['label']} </s>"
    return text_row

# process the dataframe to jsonl
def process_dataframe_to_jsonl(output_file_path, df):
    with open(output_file_path, "w") as output_jsonl_file:
        for _, row in df.iterrows():
            json_object = {
                "text": create_text_row(row),
                "label": row["label"]
            }
            output_jsonl_file.write(json.dumps(json_object) + "\n")

# load the dataset
dataset = load_dataset("ag_news", split="train")
dataset = dataset.train_test_split(test_size=0.2)
train_data = dataset["train"]
test_data = dataset["test"]

# jsonl file paths
train_json_file = "/kellogg/proj/awc6034/fine_tune/ag_news/data/train_data.jsonl"
test_json_file = "/kellogg/proj/awc6034/fine_tune/ag_news/data/test_data.jsonl"

# convert data to pandas
train_df = train_data.to_pandas()
test_df = test_data.to_pandas()

# process and save json files
process_dataframe_to_jsonl(train_json_file, train_df)
process_dataframe_to_jsonl(test_json_file, test_df)

print("===========================")
print("Data processing complete.")
print("Here is a sample of the training data:")
print(test_df.head())
print("===========================")
