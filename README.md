# Tweets_classifier_NLP

All the notebooks are implemented in GOOGLE COLAB(free cloud service and requires no setup to use specifically used for ML projects.)                

This is a getting started kaggle challenge on Natural language processing[NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/notebooks?sortBy=voteCount&group=everyone&pageSize=20&competitionId=17777)

##### Task content: build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. You’ll have access to a dataset of 10,000 tweets that were hand classified.

The dataset is available at [Kaggle](https://www.kaggle.com/c/nlp-getting-started/data)(train and test data).

> We will use various machine learning models to see how accurate they are in classifying disaster tweets.The models explored are:

* Using linear model RidgeClassifier and scikit-learn's CountVectorizer to count the words, and modelling by  EDA(Exploratory Data Analysis), Bag of Words, TF IDF, GloVe( global vectors for word representation) and implementing Baseline Model with GloVe(NLP_with_Disaster_Tweets.ipynb)

* Using BERT model(Bidirectional Encoder Representation from Transformers).Modelling it from scratch by defining helpful functions for BERT implemenation on the cleaned data.Pre-training BERT by loading  bert from tensorhub and load the tokenizer from bert layer ,then  encode and convert the data into Bert-input form and then using callbacks in training and fine tuning for learning output parameters.(BERT.ipynb)

* A approach to use transformers using the simpletransformers package that allows for Simple hyperparameter tuning, Basic k-fold cross validation, Zero pre-processing of data for prediction on MNIST classification. Also using other models like BILSTM implemented from scratch after cleaning data and the Naive Bayes algorithm by implementing from the scratch the sklearn MultinomialNB class.(simpletransformer.ipynb)

* The model implemented in notebook is based on the modelling of transformer and SVM(Support Vector Machine) after data cleaning/preprocessing  and majority voting for semanticallyidentical but wrongly labelled tweets.The Multilingual Universal Sentence Encoder is used for sentence encoding.(from tensorflow_hub).


## The models details are:

* linear model(RidgeClassifier) - Building vectors by using scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.Using cross-validation for testing  our model and see how well it does on the training data.Accuracy score roughly 0.65(isn't terrible!).

* Baseline Model with GloVe - After analysing data by EDA (with WordCloud) ,Bag of Words, TF IDF , GloVe pretrained corpus model to represent our words is used.The model is using SpatialDropout1D(0.2), one layer of LSTM(dropout=0.2) and Dense(activation='sigmoid') after adding embedding layer. Accuracy score: 0.76(pretty fine).

* BERT model- Implemented from scratch in tensorflow framwork by defining helpful functions after Data cleaning and then Pre-training BERT, finetuning it including callbacks. Accuracy score:0.845(very good) :fire:.

* Using  simpletransformers package - Using 'bert-large-uncased' in the Classification Models of simpletransformers. Applying 5-Fold Cross Validation and hypertuning it.
Accuracy score: 0.83(good) :+1:

* BiLSTM Model in Keras(Using TensorFlow backend)- After doing Exploratory Data Analysis ,followed by data cleaning , BiLSTM model is implemented from scratch using embedding matrix of glove(100d).Dropout of 0.5 is used in all layers and the optimizer is rmsprop. Callbacks consisting of reduce learning rate, checkpoint is also used in model.
Accuracy score: 0.81(not bad)

* Naive Bayes - Classification model based on Bayes Theorem.Implementation of functions like Likelihood of an entire message belonging to a particular class,Likelihood of an entire message across all possible classes using sklearn library. Accuracy score: 0.78(OK)

## Transformer+SVM+Semantically Similar Tweets:(Accuracy Score - 0.837) - The output file obtained after predicting on test data is in submission.csv file


**Pipeline of the model - Text cleaning/preprocessing(+) + Transformer + Support Vector Machine + Majority voting for semantically equivalent but mislabelled tweets + Filtering basing on keywords**

* In the training data there are many 'semantically equivalent' tweets.For example some tweets differ only in the URLs.They generate equivalence classes.

* There are classes, where tweets have 'mislabelling'. We can find 1 and 0 labels in the same class. But all tweets in such class are considered as semantically equal.

* One basic way to solve the issue is Calculate the mean for the 'target' in each class with mislabelling and the 'target' for the corresponfing records in train set is changed depending on the mean value: if mean is greater or equal 0.5 to 1, if mean is lower 0.5 to 0 .

* Using the Multilingual Universal Sentence Encoder (from tensorflow_hub) for sentence encoding. 

* Data cleaning/preprosessing to remove not very informative content for the classification.

* Sentencies are encoded in vectors and supplied for classification to Support Vector Machine by the use of "self attention mechanism".

* Using keywords for better prediction - some keywords with very high probability (sometimes = 1) signal about disaster (or usual) tweets. 

***100% accuracy is not really that possible in this problem, as far as i have seen with my models and by reading the kernels of other experts.The secret of 100% accuracy is 
simply leaked labels. The labels of the test set which is being used for estiamtion of our score is available on some other site and hence it can be used to get that 100% accuracy mark.***

***In all of the models implemented, it's AMAZING how important hyperparameters are. Try changing the learning_rate to 0.01 and see what happens. Also try changing the batch_size to 20 instead of 64. Try adding weight_decay to the optimizer functions. The accuracy of the model will be improved , but by altering some of these hyperparameters can change this "sweet spot" we found instantly.***

## References:
[NLP Getting Started Tutorial](https://www.kaggle.com/philculliton/nlp-getting-started-tutorial), 
[Prateek Maheshwari-github](https://www.kaggle.com/friskycodeur/nlp-with-disaster-tweets-bert-explained), 
[Dmitri Kalyaev Notebook](https://www.kaggle.com/dmitri9149/transformer-svm-semantically-identical-tweets), 
[Alin Cijov-github](https://www.kaggle.com/alincijov/naive-bayes-from-scratch-for-beginners/comments), 
[concrete_NLP_tutorial](https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb), 
[Kaggle notebook of szelee](https://www.kaggle.com/szelee/simpletransformers-hyperparam-tuning-k-fold-cv) 


## Support 

If you like this repo and find it useful, please consider (★) starring it (on top right of the page) so that it can reach a broader audience.
