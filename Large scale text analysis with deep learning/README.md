# Large scale text analysis with deep learning


In this project I deal with the Job Salary Prediction problem from [kaggle](https://www.kaggle.com/c/job-salary-prediction#description). 

$\textbf{Main task}$ is to predict the salary of any UK job ad based on its contents.

<img src="https://github.com/privet1mir/NLP/blob/main/Large%20scale%20text%20analysis%20with%20deep%20learning/images/salary%20prediction.png" width="600">

In this project we have both Categorical and Text features. We will learn the model on both of them. Also, we will predict not the nominal value of salary, but it's ln(1+p). It needs because we have wide range between for example ceo's salary and ordinary worker's. As it said in notebook, the distribution is fat-tailed on the right side, which is inconvenient for MSE minimization. So we work with: 

<img src="https://github.com/privet1mir/NLP/blob/main/Large%20scale%20text%20analysis%20with%20deep%20learning/images/salary.png" width="700">

And the current data that we will deal with: 

<img src="https://github.com/privet1mir/NLP/blob/main/Large%20scale%20text%20analysis%20with%20deep%20learning/images/samples.png" width="1000">

So, for our data we have: 
* Free text: Title and FullDescription
* Categorical: Category, Company, LocationNormalized, ContractType, and ContractTime
* Target: log1pSalary

The Preprocessing text data stages (which include tokenization, mapping text lines into neural network-digestible matrices and encoding the categirical data) you can see directly in the [project](), so we move directly to the deep learning part. 

## Deep learning part

The Architecture of our deep learning model folows: 

Our basic model consists of three branches:
* Title encoder
* Description encoder
* Categorical features encoder

We will then feed all 3 branches into one common network that predicts salary.

<img src="https://github.com/privet1mir/NLP/blob/main/Large%20scale%20text%20analysis%20with%20deep%20learning/images/nn_architecture.png" width="900">

For the neural network with such structure with max pooling and 64 neurons on each layer I obtained the following result: 

<img src="https://github.com/privet1mir/NLP/blob/main/Large%20scale%20text%20analysis%20with%20deep%20learning/images/learning_plot.png" width="600">

Also this project is still at work. In the future investigation I will try to implement LSTM and use different pooling methods. 
