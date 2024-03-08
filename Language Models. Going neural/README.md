# Going neural

<img src='https://raw.githubusercontent.com/yandexdataschool/nlp_course/master/resources/expanding_mind_lm_kn_3.png' width=400>

In the last project I've checked statistical approaches to language models. Now it's time to go deeper (learning). We're gonna use the same dataset as before(means corpora of [ArXiv](https://arxiv.org/) articles), except this time we build a language model that's character-level, not word level. 

The projects can be devided into three main parts: 

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

Structure of NN: 

```
        #embedding layer
        self.embed = nn.Embedding(n_tokens, emb_size)

        #conv layer
        self.conv = nn.Conv1d(in_channels=16, out_channels=hid_size, kernel_size=filter_size, stride=strides, padding=0)

        #some dense layers to map into n_tokens size + additional complexity for our nn
        self.logits = nn.Sequential(nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, n_tokens))
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

