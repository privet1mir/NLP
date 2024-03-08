# Going neural

<img src='https://raw.githubusercontent.com/yandexdataschool/nlp_course/master/resources/expanding_mind_lm_kn_3.png' width=350>

In the last project I've checked statistical approaches to language models. Now it's time to go deeper (learning). We're gonna use the same dataset as before(means corpora of [ArXiv](https://arxiv.org/) articles), except this time we build a language model that's character-level, not word level. 

The [project](https://github.com/privet1mir/NLP/blob/main/Language%20Models.%20Going%20neural/main.ipynb) can be devided into four main parts: 

## 1. Preprocessing 

Firstly we introduce special tokens: 

* Begin Of Sequence (BOS) - this token is at the start of each sequence. We use it so that we always have non-empty input to our neural network. $P(x_t) = P(x_1|BOS)$

* End Of Sequence (EOS) - you guess it... this token is at the end of each sequence. The catch is that it should not occur anywhere else except at the very end. If our model produces this token, the sequence is over.

On the next step we buildchar-level vocabulary (simply assemble a list of all unique tokens in the dataset)

Then we assign each character with its index in tokens list. This way we can encode a string into a torch-friendly integer vector.

Finally we assemble several strings in a integer matrix with shape [batch_size, text_length] by padding short sequences with extra "EOS" tokens (or cropping long sequences)

## 2. Neural Language Model

Just like for N-gram LMs, we want to estimate probability of text as a joint probability of tokens (symbols this time).

$$P(X) = \prod_t P(x_t \mid x_0, \dots, x_{t-1}).$$

Instead of counting all possible statistics, we want to train a neural network with parameters $\theta$ that estimates the conditional probabilities:

$$ P(x_t \mid x_0, \dots, x_{t-1}) \approx p(x_t \mid x_0, \dots, x_{t-1}, \theta) $$

But before we optimize, we need to define our neural network. Let's start with a fixed-window (aka convolutional) architecture:

<img src='https://raw.githubusercontent.com/yandexdataschool/nlp_course/master/resources/fixed_window_lm.jpg' width=400px>

After defining the model called `FixedWindowLanguageModel`, I also implemented categorical crossentropy over training dataset (D) function, that we want to minimize: 

$$ L = {\frac1{|D|}} \sum_{X \in D} \sum_{x_i \in X} - \log p(x_t \mid x_1, \dots, x_{t-1}, \theta) $$

Structure of Fixed window language model: 

```
FixedWindowLanguageModel(
  (embed): Embedding(136, 16)
  (conv): Conv1d(16, 64, kernel_size=(3,), stride=(1,))
  (logits): Sequential(
    (0): Linear(in_features=64, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=136, bias=True)
  )
)
```

Hyperparameters for training: 
```
batch_size = 128
lr=0.001
num of epochs = 2000
```
Training process looks like this: 

<img src='https://github.com/privet1mir/NLP/blob/main/Language%20Models.%20Going%20neural/pics/CNN_training_graph.png' width=400px>

We can see that in about 2000 epochs our model's loss reached local minima. Also we can see strong improvence in token's generation: 

Sample before training: `Bridging(Ul(^9$ś>ε-)-σHÜE+0;Ωü+vσ{qàkεb0SR:Wω;qÖτõV!ΩtN.íàρ^Bl`

Sample after training: `A Aresing in and date Fearns of clocation controm the base and models the in the rearning the model 
 MINed to the problem the comput spar graph the a network (ASNotive decision modelical and as the pro
 Applical network a section of a tectionstrar dire provel perach exportant of the proposed to extral `

The text looks similar to english, however we can do better! 

## 3. RNN Language Models

Fixed-size architectures are reasonably good when capturing short-term dependencies, but their design prevents them from capturing any signal outside their window. We can mitigate this problem by using a __recurrent neural network__:

$$ h_0 = \vec 0 ; \quad h_{t+1} = RNN(x_t, h_t) $$

$$ p(x_t \mid x_0, \dots, x_{t-1}, \theta) = dense_{softmax}(h_{t-1}) $$

Such model processes one token at a time, left to right, and maintains a hidden state vector between them. Theoretically, it can learn arbitrarily long temporal dependencies given large enough hidden size.

<img src='https://raw.githubusercontent.com/yandexdataschool/nlp_course/master/resources/rnn_lm.jpg' width=400px>

Structure of the model is the following: 

```
RNNLanguageModel(
  (embedding): Embedding(136, 16)
  (lstm): LSTM(16, 256, batch_first=True)
  (dense): Linear(in_features=256, out_features=136, bias=True)
)
```

Hyperparameters for training: 

```
batch_size = 64 
lr=0.001
num of epochs = 2000
```
Training process looks like this: 

<img src='https://github.com/privet1mir/NLP/blob/main/Language%20Models.%20Going%20neural/pics/RNN_training_graph.png' width=400px>

We can see some improvence in dev loss that fall deeply to 750 (while it was 1000 for fixed window structure). To check overall improvement we can also generate some text: 
```
Stablised Scale Senting ; We present and to exploting the matric input that detection of the active 
 A Linear Classification ; Description of underly and set of the neural networks in the image of the 
 Statistical Neural Learning ; We subjective state that internations that exploits of the the context
 Similarity ; This paper we propose a set of a diconding a set of the problem of the problem is a sea
 On Constraining 
```
We can obtain that it exactly generation of english words - what we want! 

## 4. Better sampling strategies

__Top-k sampling:__ on each step, sample the next token from __k most likely__ candidates from the language model.

Suppose $k=3$ and the token probabilities are $p=[0.1, 0.35, 0.05, 0.2, 0.3]$. You first need to select $k$ most likely words and set the probability of the rest to zero: $\hat p=[0.0, 0.35, 0.0, 0.2, 0.3]$ and re-normalize:
$p^*\approx[0.0, 0.412, 0.0, 0.235, 0.353]$.

Top k-sampling strategy is not the best one, as we know. We want diversity, but also want to receive output that we want, not random tokens with low probability.  So we can implement some improvements. 

### 4.1 Nucleus sampling

__Nucleus sampling:__ similar to top-k sampling, but this time we select $k$ dynamically. In nucleus sampling, we sample from top-__N%__ fraction of the probability mass.

Using the same  $p=[0.1, 0.35, 0.05, 0.2, 0.3]$ and nucleus N=0.9, the nucleus words consist of:
1. most likely token $w_2$, because $p(w_2) < N$
2. second most likely token $w_5$, $p(w_2) + p(w_5) = 0.65 < N$
3. third most likely token $w_4$ because $p(w_2) + p(w_5) + p(w_4) = 0.85 < N$

And thats it, because the next most likely word would overflow: $p(w_2) + p(w_5) + p(w_4) + p(w_1) = 0.95 > N$.

After implementing this we can see some improvements in result of sampling by generating some text: 

```
Efficient Spapsable Deep LD Show Asuridy Casking the Provider ; Apreasion in show semantic proposes 
 A CNN a networks features the simmary firtage between paralitions recognition is representation of a
 An Adapteld the Gielves to Models to aglost: This prediegent transferment to-based regupts ; The dat
 The Model Span Infutal Networks of Sparse Learning for Elictive as Many ; Attorm hased for they manu
 Simp-Contelliment Systemm-Transfines in Challengt Stack
```

### 4.2 Beam Search

At times, you don't really want the model to generate diverse outputs as much as you want a __single most likely hypothesis.__ A single best translation, most likely continuation of the search query given prefix, etc. Except, you can't get it.

In order to find the exact most likely sequence containing 10 tokens, you would need to enumerate all $|V|^{10}$ possible hypotheses. In practice, 9 times out of 10 you will instead find an approximate most likely output using __beam search__.

Here's how it works:
0. Initial `beam` = [prefix], max beam_size = k
1. for T steps:
2. ` ... ` generate all possible next tokens for all hypotheses in beam, formulate `len(beam) * len(vocab)` candidates
3. ` ... ` select beam_size best for all candidates as new `beam`
4. Select best hypothesis (-es?) from beam

This part also implemented through this project. We can see the following output: 

```
(Prefix: Deep): Deep Learning ; This paper, we propose a novel computational neural networks of the problem of 
```
We can see structured english text, however this also tends to converge to non-optimal tokens generation and limits the diversity of generation. So, sometimes outputs can be as follows: 
```
Transfer the problem of the problem of this paper, we propose the problem of the problem of the problem of the problem of the problem of the problem of the pro
```

At the end I need to say that we just built character-based Language Model, that can output understandable text, however, it is hard to interpret because of limitations of this model. 
