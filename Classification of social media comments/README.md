# Prohibited Comment Classification

In this project I build an algorithm that classifies social media comments into normal or toxic. Like in many real-world cases, we only have a small (10^3) dataset of hand-labeled examples to work with. We'll tackle this problem using both classical nlp methods and embedding-based approach.

<img src="https://github.com/privet1mir/NLP/blob/main/Classification%20of%20social%20media%20comments/images/banned.jpeg" width="800">

## What have I done? 


### Preprocessing and tokenization

In this part I split text into space-separated tokens using TweetTokenizer from nltk. We need this because comments contain raw text with punctuation, upper/lowercase letters and newline symbols.

### Bag of words

As a baseline I create a bag of words features in this way: 

* build a vocabulary of frequent words (for train data)
* for each training sample, I count the number of times a word occurs in it (for each word in vocabulary).
* consider this count a feature for some classifier

Also I did it by hands, without additional sklearn libraries, as you can see. 

### Naive Bayes

Now, when we have a bag of features for our samples, we can create a simple model to predict if the comments are toxic or non-poisonous. To deal with it I start with the naive bayes approach. Naive Bayes is a generative model: it models the joint probability of data. 
However it has several assumptions:
* Bag of Words assumption: word order does not matter
* Features (words) are independent given the class

  
