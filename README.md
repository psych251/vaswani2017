## Introduction

Attention is All You Need introduces an algorithm for sequence to sequence modeling that serves as the basis for much of the recent successes in language modeling (Brown 2020), image to sequence tasks (Herdade 2019), and zero shot learning in language domains (Brown 2020). The attention mechanism introduced in the paper also has been used to improve Reinforcement Learning algorithms giving them more selective resource allocation (Mott 2019). The specific results that I want to recreate are the performances for the base model in tables 2 and 3 from the paper (Vaswani 2017).


### Justification for choice of study

I am reproducing Vaswani 2017 because the Transformer algorithm introduced in the paper is foundational to my research interests. My immediate research goal is to create a computational model that is capable of interacting with multiple virtual modalities such as gesture, vision, and language comprehension, and is able to reason in an online setting. Transformers offer a viable method to incorporate multiple modalities into a single algorithm, and are the state of the art in zero-shot, pattern learning, computational systems. It is of utmost importance that I am able to recreate their results as a way of testing that my code is correct. The BLEU score from table 2 should serve as an appropriate benchmark for having accurate code.

### Anticipated challenges

To conduct this experiment, I will need to use the WMT 2014 English-German dataset. I will need to build an efficient data-pipeline that encodes the data using byte-pair encoding and feeds the data to a model for training and inference. I will need to code the multi-headed attention mechanism and from this build the encoder and decoder blocks that make up the Transformer architecture. I will need to build a project infrastructure that can combine all of these parts together for training and testing. The challenges of the project will be ensuring bug-free and scalable code that can easily be reused and re-purposed for other projects.

Additionally, finding the appropriate GPU resources will be difficult for this project. The authors report using 8 NVIDIA P100 GPUs. Obtaining access to this many GPUs may be difficult, in which case I will need to code an efficient way to perform model updates on fewer GPUs.


### Links

Project repository (on Github): [https://github.com/psych251/vaswani2017](https://github.com/psych251/vaswani2017)

Original paper: [here](https://github.com/psych251/vaswani2017/blob/master/original_paper/Vaswani-AttentionIsAllYouNeed.pdf)

## Methods

### Description of the steps required to reproduce the results

The necessary steps to reproduce the results of Vaswani et. al. are the following:

1. Download the dataset
2. Create pipeline to efficiently read dataset into project
3. Create algorithm or find appropriate library for byte-pair encoding
4. Create Transformer code base, defining the model architecture and training scheme
5. Obtain access to multiple GPUs or accumulate gradients to train on fewer GPUs
6. Validate and test trained models once complete.

### Differences from original study

The main potential difference between the environment used in the original paper and my proposed reproduction will be the available computational resources. Luckily the model architecture is defined in such a way that there are no computations that require the full training batch. This allows me to split the training batches into sub-batches for gradient computations. These gradients can be stored over multiple sub-batches which allows us to obtain gradients that are computationally equivalent to those obtained with larger batch sizes. The cost of this approach is an increase in the time it takes to train a single model. It should not change the underlying computations.

Another difference will be the way I do my random seeding. I do not believe Vaswani et. al. report their random seeds. And even if they did, even slight differences in the code structure could lead to different random sampling. This will inherently make the reproducibility stochastic.

Another difference is the libraries used for automatic differentiation (and other computations). Vaswani et. al. indicate that they used tensorflow whereas I am using pytorch.

## Results

### Data preparation

See [this script](https://github.com/psych251/vaswani2017/blob/master/transformer/transformer/datas.py) for the code that performs the data preparation.

### Key analysis

Bleu scores were evaluated on the [2014 newstest](https://nlp.stanford.edu/projects/nmt/).

![Bleu Score Comparison](./bleu_scores.png)


## Discussion

### Summary of Reproduction Attempt

It appears that this reproducibility project was successful. The goal was to train a single model that could match or beat the reported result of 0.273. The model was actually so successful, with a bleu score of 0.329, that it makes me worried that my evaluation contains a bug. I could not find one, however.

Some lingering differences that may explain the improved result is that I used the WMT English to German 2014 dataset found [here](https://nlp.stanford.edu/projects/nmt/) which may have been different than the one used in the paper. It is worth noting that I found the nearly 30 sentence pairs in the dataset that were english to english rather than english to german. It seemed as though the quantity of these pairs was likely negligible givien my results.

Additionally, after I had trained the model that I reported, I noticed that I had added an extra normalization layer in the encoding and decoding layer definitions that was not included in the Vaswani et. al. description. This may have further improved results.

Another difference worth noting is that I never implemented label smoothing. It would appear that it was not an important aspect of the training. Although I would be interested to see how the model improves from label smoothing.

### Commentary

I was surprised by how important parallelization of the GPUs was. The models took an extremely long time to train even with an efficient multi-processing algorithm. I am curious if the NVIDIA p100 GPUs are significantly faster than the TITAN X GPUs I was using.
