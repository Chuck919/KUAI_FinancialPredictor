# KUAI_FinancialPredictor
Uses AI sentiment analysis and market data to predict whether you should buy or sell an asset. Also has potential for trading bot implementation

# Steps

## 1. Download Requirements
  First download the requirements.txt file. You can do that by downloading the file and running     the code here:
```
pip install -r requirements.txt
```
  
## 2. Train the Models
  There are 2 models, Bernoulli Naive Bayes (BNB) and Linear Support Vector Classification         (LinearSVC) models. I have attached an explanation of each below. There already exists a trained version of both, using the Project_Data.csv. You can do this by updating what data you would like in the Project_Data.csv file (however this is just a small example file, you can find the dataset on [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)) following the same format (or just simply use that file). The models will then be saved to their respective files as well as the vectorizer in a tfidf_vectorizer.joblib file. Typically, both models have a precision rate of 80-85%.
  
### Bernoulli Naive Bayes (BNB): 
  Bernoulli Naive Bayes is a classification algorithm commonly used in text analysis, making it a valuable tool for tasks like sentiment analysis. It operates on the principle of Bayes' theorem, leveraging the assumption that features are conditionally independent given the class. In the context of text, it's particularly effective when dealing with binary features, like whether specific words are present or absent. This simplicity and efficiency make BNB well-suited for scenarios where the goal is to categorize documents into different classes based on the occurrence of specific features.

### Linear Support Vector Classification (LinearSVC): 
  Linear Support Vector Classification is a powerful algorithm used for linear classification problems, including text classification. It works by finding the optimal hyperplane that separates different classes in a high-dimensional space. In simpler terms, it's adept at drawing a line (or hyperplane) in the data to distinguish between categories. For text classification tasks, where the goal is to categorize documents, LinearSVC excels in scenarios where the data can be effectively separated by a straight line. Its versatility and efficiency make it a popular choice, especially when dealing with large-scale text datasets common in natural language processing projects.

## 3. Make New Predictions
  After training the models, you can then access the NewPredictions.py file, which uses api.mediastack.com to gather news articles (you may need to create your own access key for this), or you can simply comment out the top section of the code to use the news_data.csv file as it is. After running, it will then place either a value of 4 (positive sentiment), or a value of 0 (negative sentiment) by each article and save those predictions to the predictions.csv file.

## 4. Placing Orders
  After the predictions.csv file has been created (or you can use the existing predictions.csv file), you can then run the TradingBot.py file, which will then only take the predictions that both the BNB and SVC model agree on and determine whether the majority of the sentiment is positive or negative and return a BUY or SELL signal depending on the sentiment. You can also expand on the code at the very bottom of the file which uses the Binance-Python library to automatically create buy and sell orders on your behalf. However, because the goal of this project was not to create a trading bot, that functionality is very basic.

#### This project was created by @aliceKuang12 and the KU AI Financial Predictor Team
