# Prohibited Comment Classification

In this [project](https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/Classification_of_comments.ipynb) I build an algorithm that classifies social media comments into normal or toxic. Like in many real-world cases, we only have a small (10^3) dataset of hand-labeled examples to work with. We'll tackle this problem using both classical nlp methods and embedding-based approach.

<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/ban_hammer.png" width="550">

## What have I done? 


### Preprocessing and tokenization

We have the data which looks like that: 

<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/comments.png" width="500">

To begin with I split text into space-separated tokens using TweetTokenizer from nltk. We need this because comments contain raw text with punctuation, upper/lowercase letters and newline symbols.


### Bag of words

As a baseline I create a bag of words features in this way: 

* build a vocabulary of frequent words (for train data)
* for each training sample, I count the number of times a word occurs in it (for each word in vocabulary).
* then I consider this count as a feature for some classifier

Also I did it by hands, without additional sklearn libraries, as you can see. 

As a result we have BOW dictionary (it's shape is **num of samples x size of word's dict** ) for our train samples like this: 

<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/bow_dict.png" width="500">

### Naive Bayes

Now, when we have a bag of features for our samples, we can create a simple model to predict if the comments are toxic or non-poisonous. To deal with it I start with the naive bayes approach. Naive Bayes is a generative model: it models the joint probability of data. 

However it has several assumptions:
* Bag of Words assumption: word order does not matter
* Features (our words) are independent for distinct class

I implement this method in **BinaryNaiveBayes class** with fit and predict methods. The underlying theory for the naive bayes you can see below: 

The basis of the method is that we can rewrite the probability of each class with defined set of features by the Bayes formula: 

$$\large{\mathcal{P}(y=k|x) = \frac{\mathcal{P}(x|y=k) \cdot \mathcal{P}(y=k)}{\mathcal{P}(x)}}$$
  
So the main task of our **generative** model is to find such k that maximizes the probability of meeting our features x: 

$$\large{y* = arg \underset{k}{max} \frac{\mathcal{P}(x|y=k) \cdot \mathcal{P}(y=k)}{\mathcal{P}(x)}  = arg \underset{k}{max} \mathcal{P}(x|y=k) \cdot \mathcal{P}(y=k) = arg \underset{k}{max}  \mathcal{P}(x, y=k)}$$ 

More clearly this formulates in perfect [course](https://lena-voita.github.io/nlp_course/text_classification.html) by Lena Voita. 

So, our main goal is to obtain the joint probability of our data.

How to define parts of the formula above? 

$\large{\mathcal{P}(y=k) = \frac{N(y=k)}{\sum_i{N(y=i)}}}$ - it is just the proportion of documents with the label $k$. 

Then we need to define the term $\mathcal{P}(x|y=k)$ and there is where the "naive" is! 

We can calculate it using the assumption of our features independends: $\large{\mathcal{P}(x|y=k) = \mathcal{P}(x_1,x_2,...,x_n|y=k) = \prod_i{\mathcal{P}(x_t|y=k)}}$, where each term of product can be calculated as: $\large{\mathcal{P}(x_i|y=k) = \frac{N(x_i, y=k)}{ \sum_{t}  N(x_t, y=k)}}$, where the sum is over the set of all possible features. 

To avoid the 0-probability of unique words appearance, we'll just add small regularization $\delta$ (which called Laplce-smoothing, I use $\delta = 1$). So we have: 

$\large{\mathcal{P}(x_i|y=k) = \frac{N(x_i, y=k) + \delta}{ \sum_{t}  N(x_t, y=k) + \delta |V|}}$, where $\large{|V|}$ - the size of our vocabulary. 

Finally we can compute the log probabilities (log because summarise is easier numerically and better because of our small probability values): 

$$\large{\log{\mathcal{P}(x, y=k)} = \log{\mathcal{P}(y=k)} + \sum_i {\log{\mathcal{P}(x_t, y=k)}}}$$

<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/bow_auc.png" width="500">

The final accuracy for Naive Bayes algorithm is pretty well, but we can better. 

### Logistic Regression 

We can use the logistic regression from the box to compare the result with naive bayes. I tuned just one parameter - inverse of regularization strength and set it at $\large{C = 0.2}$. The final ROC-AUC and accuracy you can see below: 

<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/logreg_auc.png" width="500">

Even without any tunning the log reg has better result, but just a little bit. 

### TF-IDF

We can move deeper and prioritize rare words (which means more important in some cases) by tf-idf implementation. 

$$\large{feature_i = { Count(word_i \in x) \times { log {N \over Count(word_i \in D) + \alpha} }}}$$ where x is a single text, D is your dataset (a collection of texts), N is a total number of documents and $\large{\alpha}$ is a smoothing hyperparameter (typically 1). 
And $\large{Count(word_i \in D)}$ is the number of documents where $\large{word_i}$ appears.

Also I normalize each data sample after computing tf-idf features.

After implementation this methoda (also by hands without any additional packages) we can see some improvement in our ROC-AUC and accuracy, which means that our algorithm works: 

<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/tf-idf_auc.png" width="500">

### Word Vectors

And now we can implement the most powerful approach in this project - word vectors. Instead of counting per-word frequencies, we shall map all words to pre-trained word vectors and average over them to get text features.

This should give us two key advantages: (1) we now have 10^2 features instead of 10^4 and (2) our model can generalize to word that are not in training dataset.

I load the pre-trained model from gensim, apply it to our features and trained the logistic regression on them: 

  ```
  import gensim.downloader 
  embeddings = gensim.downloader.load("fasttext-wiki-news-subwords-300")
  ```
<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/vec_auc.png" width="500">

We can see a large improvement in accuracy score and also test_auc. Which means that while the classic approaches are interesting and sometimes useful, now is the time of modern approaches of pre-trained libraries. 

### Final Results

Final AUC graphs and test accuracy you can see below 

<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/final_auc.png" width="500">

| Model | Accuracy  |
| ------- | --- |
| Naive Bayes | 0.756 | 
| LR | 0.772 | 
| LR + TF-IDF | 0.790 | 
| LR with WV | 0.884 | 

Of course we can improve such result in different ways. We can:
* train embeddings from scratch on relevant (unlabeled) data
* multiply word vectors tf-idf
* concatenate some of embeddings

But maybe next time:) 
