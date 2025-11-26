# Modeling Stock Market Movements Using Twitter Sentiment and Machine Learning: A Case Study on Donald Trump’s Tweets
Proyecto de Titulación - Modelado Impacto del Sentimiento expresado en Twitter con el Mercado Bursatil

## Overview
This project investigates whether sentiment extracted from Donald Trump’s Twitter posts can improve the prediction of daily directional movements of the S&P 500.
The workflow integrates:

- Natural Language Processing (NLP) for tweet cleaning, tokenization, lemmatization, sentiment scoring (VADER, FinBERT), and Word2Vec embeddings
- Market features derived from the S&P500
- Machine Learning models: Logistic Regression, XGBoost, AutoGluon
- Explainability: SHAP
- Time-series validation: using TimeSeriesSplit

This pipeline is designed to evaluate whether social sentiment acts as a complementary signal to traditional financial variables.

## Data Sources

- https://www.thetrumparchive.com : 56,571 tweets were extracted from 2009/05/04 to 2021/01/08 while the following features: id, text, date, favorites, retweets, isRetweet, device
- Yahoo Finance API: Market Data relating S&P500 containing the following features: date, open, close, volume, high and low

## Project Pipeline

### Data Preprocessing
- Tweet cleaning (URLs, mention, hashtag removal)
- Tokenization, lemmatization (NLTK)
- Sentiment Scoring (VADER and FinBERT)

### Word2Vec Embedding
- Train Word2Vec (Vector Size = 200)
- Create tweet-level embeddings (mean of word vectors)
- Expand embeddings into 200 columns

### Aggregate Stock Market Data
-Download SPY data using Yahoo Finance
- Merge tweet dataset with market dataset
- Create the supervised leaerning target variable

### Modelling
- Generate a logistic regression pipeline as a baseline model with polynomial features, standard scaler and selectKBest
- Train AutoGluon TabularPredictor, evaluate it using precision and retrieve a leaderboard and model parameters
- Model and Optimize a XGBoost pipeline
- Perform a statistical comparison of the models using the Wilcoxon test and the t-test.

### Explainability
- Calculate SHAP values for XGBoost model
- Produce summary plot, beeswarm plot and waterfall plot

## Setup

In order to replicate this experiment make sure to have the following libraries: numpy, pandas, matplotlib, yfinance, nltk, scikit-learn, seaborn, shap, autogluon.

## Running the project

- Open the following notebook on google colab:  https://colab.research.google.com/drive/1_HNRZ5kXzANLqahj75ft2XI4IfV4dZoS?usp=sharing
- Ensure that your runtime environment is properly configured to leverage GPU acceleration.
- Proceed through the notebook step by step, each cell is structured for sequential execution.
