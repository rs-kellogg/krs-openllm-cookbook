# Example Use Case: 10K Processing

Here's an example application to show you some aspects of what we've learned so far. Let's look at a very large collection of financial documents - 10-K reports. A comprehensive report filed annually by all publicly traded companies. There is __a lot__ of information in these reports, and they can be very long.

Note: We have all of them available on KLC

We can imagine lots of things we might want to do with these reports, but to keep things simple lets just imagine we want to summarize one section: MD&A (Management Discussion and Analysis). (note: missing rigorous evaluation, previous workshop showed one with an information extraction problem)

> Management discussion and analysis (MD&A) is a section of a public company's annual report or quarterly filing. The MD&A addresses the company’s performance. In this section, the company’s management and executives, also known as the C-suite, present an analysis of the company’s performance with qualitative and quantitative measures. https://www.investopedia.com/terms/m/mdanalysis.asp

## Find a Model

Let's go on the Hugging Face Model Hub and find an appropriate model.

## Create a Summarization Pipeline

Now let's write some python code to do the work. Note how simple it is! This is the easy vanilla case, we aren't measuring performance, fine-tuning, or using any RAG techniques.

## Output

Example summary. Show video.
