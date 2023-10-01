# Multilingual Embedding-based Machine Translation


In this project I create the simple UK-RU word-based translator and translated the fairy tale from  ukranian language into russian. 

$\textbf{Attention!}$ In this project you can see machine translation system without using parallel corpora, recurrent neural network and such wonderful things.

## Step 1. Embedding space mapping

Let $x_i \in \mathrm{R}^d$ be the distributed representation of word $i$ in the source language, and $y_i \in \mathrm{R}^d$ is the vector representation of its translation. Our purpose is to learn such linear transform $W$ that minimizes euclidian distance between $Wx_i$ and $y_i$ for some subset of word embeddings. Thus we can formulate so-called Procrustes problem:

$W^*= \arg\underset{W}{min} \underset{i}{\sum}||Wx_i - y_i||_2$

or

$W^*= \arg\underset{W}{min}||WX - Y||_F$

where $||*||_F$ - Frobenius norm.

![embedding_mapping.png](https://github.com/yandexdataschool/nlp_course/raw/master/resources/embedding_mapping.png)

## Step 2. Making it better (orthogonal Procrustean problem)

It can be [shown](https://arxiv.org/pdf/1702.03859.pdf) that a self-consistent linear mapping between semantic spaces should be orthogonal.
We can restrict transform $W$ to be orthogonal. Then we will solve next problem:

$W^*= \arg\underset{W}{min}||WX - Y||_F \text{, where: } W^TW = I$

$I \text{- identity matrix}$

Instead of making yet another regression problem we can find optimal orthogonal transformation using singular value decomposition. It turns out that optimal transformation $W^*$ can be expressed via SVD components:

$X^TY=U\Sigma V^T$, singular value decompostion}

$W^*=UV^T$

## Results

My model has a valid precision and perfectly solve the easy examples, as you can see: 

![simple_reuslts](https://github.com/privet1mir/NLP/blob/main/Multilingual%20Embedding-based%20Machine%20Translation/results_simple_expr.png)

However we don't work with the form of the words, so we can see problems with words such as 'смiявся' (such key not founded). The model can translate the simple form 'сміятися', but not more. 

Eventually these problems can be solved by using language models. In later projects:) 

Jpynb code and final result of fairy tale translation you can see in the repository. Check rus_translated_fairy_tale txt.  
