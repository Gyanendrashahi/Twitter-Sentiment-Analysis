# Twitter-Sentiment-Analysis

This script can tell you the sentiments of people regarding to any events happening in the world by analyzing tweets related to that event. It will search for tweets about any topic and analyze each tweet to see how positive or negative it's emotion is. You might want to check out this complete text and video based detailed 

## Getting Started
 
First of all login from your Twitter account and goto [Twitter Apps](https://apps.twitter.com/). Create a new app ([How to create twitter app](http://www.letscodepro.com/twitter-sentiment-analysis/)) and goto __Keys and access tokens__ and copy Consumer Key, Consumer Secret, Access Token and Access Token Secret. We will need them later. 

### Installation

Download or Clone the repo, Navigate to the directory containing the files and run
```
python setup.py install
```
or if you have different versions of python installed then
```
python3 setup.py install 
```
to install the dependencies.


### Usage

Once you have created an app on twitter and installed all the dependencies by running __setup.py__, open main.py and paste your Consumer Key, Consumer Secret, Access Token and Access Token Secret. After that save and run the script. You will be prompted to enter the keyword/hashtag you want to analyze and the number of tweets you want to analyze. Once the analysis is completed, a pie chart will be generated disclosing the results of analysis.

## Built With

* Python 3.6
* tweepy
* textblob
* matplotlib

## Contributing

1. Fork it
2. Create your feature branch: git checkout -b my-new-feature
3. Commit your changes: git commit -am 'Add some feature'
4. Push to the branch: git push origin my-new-feature
5. Submit a pull request

## Authors

Gyanendra Shahi

## overall

 sentiment analysis project into its components and the tools used for each purpose:

                            Data Collection:
Purpose: Gather a large dataset of text containing various sentiments to train the sentiment analysis model.
            Tools:
Twitter API: Accesses Twitter's data to collect tweets related to the topic of interest.
Web scraping libraries like BeautifulSoup or Scrapy: Extract text data from websites, forums, or other online sources.

                            Data Preprocessing:
Purpose: Clean and prepare the collected data for analysis.
            Tools:
    NLTK (Natural Language Toolkit) or spaCy: Perform tasks like tokenization, removing stop words, and lemmatization.
Regular expressions: Pattern matching to remove URLs, special characters, and other noise from text data.
                            Feature Extraction:
Purpose: Convert text data into numerical or vectorized representations suitable for machine learning models.
            Tools:
CountVectorizer or TfidfVectorizer from scikit-learn: Convert text data into bag-of-words or TF-IDF matrices.
Word embeddings: Pre-trained models like Word2Vec, GloVe, or FastText to represent words as dense vectors.
                            Model Training and Evaluation:
Purpose: Train machine learning or deep learning models to predict sentiment from text data and evaluate their performance.
            Tools:
Machine learning algorithms: Naive Bayes, Support Vector Machines (SVM), Logistic Regression, etc., implemented in scikit-learn.
Deep learning frameworks: TensorFlow or PyTorch to build and train neural network models like LSTM (Long Short-Term Memory) or CNN (Convolutional Neural Network).
Evaluation metrics: Accuracy, precision, recall, F1-score, confusion matrix, etc., to assess model performance.
                            Model Deployment:
Purpose: Deploy the trained model to make predictions on new text data.
            Tools:
Flask or Django: Web frameworks to create APIs or web applications for model deployment.
Cloud platforms: Services like AWS, Azure, or Google Cloud for hosting and scaling deployed models.
Docker: Containerization tool for packaging the model and its dependencies into a portable unit.
                            Visualization and Reporting:
Purpose: Present the results of sentiment analysis in a clear and understandable format.
            Tools:
Matplotlib or Seaborn: Python libraries for creating visualizations like bar charts, pie charts, or line plots.
Word clouds: Visualization technique to display the most frequent words in the analyzed text data.
Dash or Streamlit: Frameworks for building interactive web-based dashboards to visualize sentiment analysis results.
