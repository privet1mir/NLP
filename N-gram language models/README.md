# N-gram language models or how to write scientific papers


<img src="https://cdn.theatlantic.com/thumbor/swh75pApLfNnXOBu2O6BhrSpl6o=/0x0:4800x2700/960x540/media/img/mt/2021/05/science_publishing_is_a_joke/original.jpg" width="700">

In this [project](https://github.com/privet1mir/NLP/blob/main/N-gram%20language%20models/N_gram_language_models.ipynb) we will train our language model on a corpora of ArXiv articles and see if we can generate a new one!

The projects can be devided into three main parts: 

## 1. N-Gram Language Model

A language model is a probabilistic model that estimates text probability: the joint probability of all tokens $w_t$ in text $X$: $P(X) = P(w_1, \dots, w_T)$.

It can do so by following the chain rule:
$$P(w_1, \dots, w_T) = P(w_1)P(w_2 \mid w_1)\dots P(w_T \mid w_1, \dots, w_{T-1}).$$

The problem with such approach is that the final term $P(w_T \mid w_1, \dots, w_{T-1})$ depends on $n-1$ previous words. This probability is impractical to estimate for long texts, e.g. $T = 1000$.

One popular approximation is to assume that next word only depends on a finite amount of previous words:

$$P(w_t \mid w_1, \dots, w_{t - 1}) = P(w_t \mid w_{t - n + 1}, \dots, w_{t - 1})$$

Such model is called __n-gram language model__ where n is a parameter. For example, in 3-gram language model, each word only depends on 2 previous words.

$$
    P(w_1, \dots, w_n) = \prod_t P(w_t \mid w_{t - n + 1}, \dots, w_{t - 1}).
$$

## 2. Evaluating language models: perplexity 

Perplexity is a measure of how well your model approximates the true probability distribution behind the data. __Smaller perplexity = better model__.

To compute perplexity on one sentence, use:
$${\mathbb{P}}(w_1 \dots w_N) = P(w_1, \dots, w_N)^{-\frac1N} = \left( \prod_t P(w_t \mid w_{t - n}, \dots, w_{t - 1})\right)^{-\frac1N},$$


On the corpora level, perplexity is a product of probabilities of all tokens in all sentences to the power of $1/N$, where $N$ is __total length (in tokens) of all sentences__ in corpora.

This number can quickly get too small for float32/float64 precision, so we can firstly compute log-perplexity (from log-probabilities) and then take the exponent.

$$
    {\ln{\mathbb{P}}}(w_1 \dots w_N) = \ln \left( \prod_t P(w_t \mid w_{t - n}, \dots, w_{t - 1})\right)^{-\frac1N} = - \frac{1}{N} \sum_t \ln{P(w_t \mid w_{t - n}, \dots, w_{t - 1}})
$$

## 3. LM Smoothing

The problem with our simple language model is that whenever it encounters an n-gram it has never seen before, it assigns it with the probabilitiy of 0. Every time this happens, perplexity explodes.

To battle this issue, there's a technique called smoothing. The core idea is to modify counts in a way that prevents probabilities from getting too low. The simplest algorithm we can implement is Additive smoothing (aka Lapace smoothing):

$$ P(w_t | prefix) = { Count(prefix, w_t) + \delta \over \sum_{\hat w} (Count(prefix, \hat w) + \delta) } $$

If counts for a given prefix are low, additive smoothing will adjust probabilities to a more uniform distribution. Not that the summation in the denominator goes over _all words in the vocabulary_.

The results of our hand-made language model you can see in [notebook](https://github.com/privet1mir/NLP/blob/main/N-gram%20language%20models/N_gram_language_models.ipynb). We can't talk about precision, cause it is just fun model that generate hardly-interpretable text. 
