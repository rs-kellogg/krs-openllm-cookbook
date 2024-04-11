# Introduction

Welcome to the workshop on using open-source Large Language Models (henceforth: LLMs) at Kellogg. Some of you may have attended our last workshop on using OpenAI. This one shares some themes, but focuses on isses that are relevant to getting open source models to run effectively using on-prem resources. We will show you how to select an open source model, run it on Northwestern Graphics Procssing Units (GPUs), and how to improve peformance both through offline training and online augmentation with external resources. To get the ball rolling, let's take a quick look at the following project lifecycle diagram.

__Project Lifecycle__

This diagram outlines the lifecycle of a machine learning (ML) project, particularly one involving a Large Language Model (LLM) such as GPT. It is divided into three main stages, each with specific steps:

* __Scope__: Define the use case: This is the initial step where the goals, requirements, and problems to be solved by the ML project are clearly defined.
Select

* __Select__: Choose an existing model or pretrain your own: Here, a decision is made whether to utilize an existing pre-trained model or to pretrain a new model from scratch, based on the requirements of the use case. We will not discuss traning your own model from scratch, for most of us that is beyond our means. Instead, we'll focus on selecting an existing open source model and adapting it to our needs.

* __Adapt and Align Model__: Once we have a model, we have to make it do what we want. At this phase of the process, our goal is to make sure our performance is as good as possible for our use case. These are the steps we take:
    * Prompt engineering: Crafting the inputs (prompts) to the model in such a way that the desired output is more likely to be generated.
    * Fine-tuning: Our chosen model is further trained (fine-tuned) on a specific dataset to better perform on tasks related to the use case.
    * Evaluate: The model's performance is assessed to ensure it meets the defined goals (can't stress enough how useful this is).
    * Align with human feedback: Human judgments are used to guide the model's outputs, ensuring they align with human values and expectations. We won't be doing this in this workshop, it is again probably beyond most people's means to train a new reward model from scratch using human feedback. However, note that there are existing reward models (e.g., to de-toxify output) that would not require too much work to use.

* __Application Integration__: Now our model is adapted, it must be deployed. Two things we focus on here: optimizing for model size, and extending a model's knowledge by integrating external resources at run-time.
    * Optimize and deploy model for inference: The model is optimized for performance (speed and accuracy) and deployed in a production environment where it can be used to make predictions or generate text.
    * Augment model and build LLM-powered applications: The model is potentially augmented with additional capabilities, and applications that leverage the power of the LLM are built around it.

Each of these stages is crucial for the successful integration of an LLM into a practical application, and the process is iterative, often requiring multiple cycles through these stages to refine the model and its application.

## Define the Use Case

Your plan should specify:

* What data will I be using to achieve my research goal?
* How much data do I need?
* How will I evaluate LLM output?
* What counts as good enough?

We won't go more into this phase of the lifecycle here, but just be aware that everything else, form model choice to data collection follows from getting this part right.

Your choice of open source model is constrained by the type of use case you are implementing, for example a translation task will use a sequence2sequence architecture and correpsonding model.

## Select a Model

There is no question that proprietary models currently have an edge in many areas of performance. For example, the current Chatbot leaderboard [scroll down and show screenshot] is dominated by closed-source, proprietary systems.

Why then, would you choose to use an open source LLM? There are a variety of reasons:

- __Reproducibility__: propietary models can change at any time
- __Data privacy__: if you use private knowledge sources for fine-tuning or RAG, you may not want or even be allowed to upload to proprietary models
- __Flexibility to adapt a model__: While some fine-tuning possibilities exist, you have more flexibility with open source packages
- __Ability to share a model__: If you modify an open-source model with fine-tuning, you can share it however you'd like, for example for open science reasons
- __Cost at inference time__: Cost may or may not be an issue, but we have sunk costs at Northwestern in terms of GPU nodes and there is no marginal cost for you to use them. May also reduce latency.

Models vs. Code: It's important to understand this difference when you start using open source models. Models are very large files that store billions of weights (floating point numbers), intended to be used for a particular (transformer) model architecture. Code is available to train and run these models

* Models are frequently obtained from model hubs
* Code is obtained via python package managers, github, etc.
* The Hugging Face model hub is widely used
* You should use the nature of your task, and also look at leaderboards to select your particular model.
* Aside: once we have an open source model, we're going to have to figure out how to run it locally. For that, we'll often need GPUs in order to get any kind of reasonable performance, and that will be the topic of our next section: Runing LLMs on Northwestern GPUs


## Adapt the Model

We saw above that there are reasons to use open source LLMs, even though proprietary models generally have higher out-of-the-box performance on many tasks. Here we discuss ways to remedy this situation for your particular use case. 

As a first step however, we recommend that you always try to create the best possible prompting strategy to elicit the behavior that you want. This is a low-cost, minimal expertise approach that can often improve performance. Some strategies here include few-shot learning (providing examples of what you want), and chain-of-thought (breaking down a task into incremental steps, like solving a word problem as a sequence of logical deductions).

Once you've exausted prompt engnieering, it could very well be time to try something else. In this section, we focus on fine-tuning, which is a way to adapt the weights of the model to generate output that is more aligned with your use case. There are a variety of strategies here, but the core of all of them is that you provide the LLM with input/output pairs of what you want, and the model will adapt its weights to match desired outputs more closely.

Fine-tuning results in a new model that is based on the old model. If all works well, the new model has two core advantages: (1) improved accuracy and relevance of output, and (2) reduced needs for elaborate prompting (and thus saving your context window).

If you want to measure accuracy, you're going to have to figure out which evaluation metric to use. For classification and information extraction tasks, you can count true/false positives/negatives, and calculate precision, recall, etc.

Ambreen will be your guide on fine-tuning later in this workshop.

## Application Integration

Once you've selected a model, and potentially adapted it through fine-tuning, you will need to deploy it as an application. How this works depends on what you're trying to do, but here we'll focus on two things that may be relevant to a variety of applications.

First, a quick word about compute resources. Big models require big resources. A standard 70-b parameter model will not fit into the memory of a single A100 GPU node. There are some strategies to help tackle this, such as sharding a model across multiple GPUs, but you can also use something called "quantization" to reduce memory footprint. Quantization maps full precision floating point numbers to smaller precision numbers that require less memory. Such "quantized" versions of models may allow you to run even a 70b parameter model on a single A100 GPU. Ambreen will provide more details later on.

Second, it is important to note that an LLM does not have to stand on its own, but can be integrated with other resources. Many of you have probably already heard of "Retrieval Augmented Generation", or "RAG" for short. This is a technique to use a prompt to lookup relevant resources in a knowledge base (such as a document collection), use these resources to augment a prompt, and feed the augmented prompt to the LLM. This is one way to defeate the "training knowledge cut-off" problem, where an LLM only knows what it was trained on up to a certain date. It can also result in fewer "hallucinations", improving the fidelity and factualness of replies.

Peilin will be your guide on RAG later in this workshop.
