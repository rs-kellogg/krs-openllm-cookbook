###########################
#  Process Annual Reports #
###########################

# ------------------------------------------------------------------------------
# Package Imports
# ------------------------------------------------------------------------------

from datasets import Dataset
from transformers import pipeline
from pathlib import Path
import pandas as pd
import torch
import re
import os


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
def extract_mda(text):
    mda_text = None

    # obtain the second occurrence of "Discussion and Analysis of Financial Condition" with wildcards
    pattern = r"Discussion[\s,.-]*and[\s,.-]*Analysis[\s,.-]*of[\s,.-]*Financial[\s,.-]*Condition"
    mda_matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if len(mda_matches) >= 2:
        m = mda_matches[1]
        mda_text = text[m.end() :]
        return " ".join(mda_text.split()[:250])
    return mda_text


# ------------------------------------------------------------------------------
def main(
        cache_dir: Path = Path("/projects/kellogg/.cache"),
        input_dir: Path = Path("/kellogg/data/EDGAR/10-K/2023"),
        output_file: Path = Path("/projects/kellogg/output/annual_report_output.csv"),
        model_checkpoint: str = "Falconsai/text_summarization",
        num_files: int = 10
    ):

    # validate input parameters
    assert cache_dir.exists() and cache_dir.is_dir()
    assert input_dir.exists() and input_dir.is_dir()
    assert num_files > 0
    
    os.environ["HF_HOME"] = str(cache_dir)

    # get listing of 10K files
    files = list(input_dir.glob("*.txt"))[:num_files]
    files.sort()

    # load and clean text, extr
    data_dict = {"doc": [], "text": []}
    for f in files:
        print(f"loading: {f.name}")
        text = clean_html(f.read_text())
        mda_text = extract_mda(text)
        if mda_text is None:
            continue
        data_dict["doc"].append(f.name)
        data_dict["text"].append(mda_text)

    # create a dataset object
    dataset_10k = Dataset.from_dict(data_dict)
    print(f"created dataset: {dataset_10k}")

    # apply summarization pipeline to dataset
    summarizer = pipeline("summarization", model=model_checkpoint)
    dataset_10k = dataset_10k.map(
        lambda batch: {
            "summary": summarizer(
                batch["text"],
                max_length=50,
                min_length=30,
                do_sample=False,
                truncation=True,
            )
        },
        batched=True,
    )

    # output to file
    dataset_10k.to_csv(output_file)


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using CUDA: {torch.cuda.is_available()}")
    main()
