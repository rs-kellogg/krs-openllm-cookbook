# Using GPUs at Northwestern

Now we're going to discuss how you execute a model on on-premise resources, particularly on GPUs. In order to use the code shown in this workshop, you will need to be a member of a Quest allocation. Most of you have been added to one for this workshop - "e32337". If you have not been (or don't know how to find out if you are, please let us know and we'll help you out. This allocation will be avaialble for several months for you to use.

Note there are other ways to get access to GPUs, as shown here. For trying stuff out, I particularly like Google Colab, it's free for lower tier GPUs, or for $10/month you can even get access to an Nvida A100 or V100.

## Parallel Computing for LLMs

The first question to ask is why use GPUs? Why not good old fashioned CPUs? The answer here is speed, with GPUs several of order magnitude faster than CPUs. And the reason for that is parallelism.

CPUs are miracles of engineering. They execute complex instructions at blindingly fast speeds. However, a single core CPU can only run one instruction at a time. They do so very quickly, and they can swap in and out different tasks to make it appear they are running them in parallel, but they aren't in fact doing so.

To overcome this basic fact of nature, most computers now ship with "multi-core" CPUs, where every core can run a job in parallel with the others. For example, the latest generation of KLC nodes has 64 cores, meaning 64 actually parallel jobs.

This is where GPUs come in. GPUs were developed (obvsiously) for displaying graphics, which involves vast number of vector and matrix operations on floating point numbers, much of which can be executed in parallel. It turns out this is exactly what is needed for training and running neural network deep learning models, and they were borrowed for this purpose. An Nvidia A100 GPU has several thousand cores, and the H100 chip has around 15,000!

Many of you will have heard of something called "CUDA", and maybe have some vague ideas about its relationship to GPUs. CUDA is a software platform created by Nvida to enable software engineers to use Nvidia GPUs, putting objects in its memory, controlling parallism, getting the output back into normal RAM. Most of us will never touch CUDA directly, as shown in this software stack. However, can be good to know which version of CUDA is supported on whatever machine your using, as the installed software layers up this stack must be configured appropriately.

## Sample Python GPU Code

Here is some sample Python code to detect whether or not CUDA is available, and which version is available. Note that this code will run on both GPU and non-GPU machines. But if you're running it without a GPU, the tensor code will run thousands of times slower!

## Slurm Script to Access GPU Nodes

The Northwestern Research Computing team has purchased dozens of Nvidia-based GPUs and made them available through the Quest HPC cluster. You can use the Quest allocation mentioned above to try them out. Quest GPU nodes are accessible though the "Slurm" scheduling software. Using Slurm, you write scripts that will be submitted to be executed on GPU nodes when the necessary resources are available to be used.

Let's take a look at a bash script that will execute the Pytorch GPU test python code we saw above. This looks pretty much like any old bash script, but there are some directives at the time in the form of comment lines that tell Slurm what resources are being requested, and where to generate output. Let's break down the script.

