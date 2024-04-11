# Using GPUs at Northwestern

Now we're going to discuss how you execute a model on on-premise resources, particularly on GPUs. In order to use the code shown in this workshop, you will need to be a member of a Quest allocation. Most of you have been added to one for this workshop - "e32337". If you have not been (or don't know how to find out if you are, please let us know and we'll help you out. This allocation will be avaialble for several months for you to use.

Note there are other ways to get access to GPUs, as shown here. For trying stuff out, I particularly like Google Colab, it's free for lower tier GPUs, or for $10/month you can even get access to an Nvida A100 or V100.

## Parallel Computing for LLMs

The first question to ask is why use GPUs? Why not good old fashioned CPUs? The answer here is speed, with GPUs several of order magnitude faster than CPUs. And the reason for that is parallelism.

## Sample Python GPU Code

## Slurm Script to Access GPU Nodes



