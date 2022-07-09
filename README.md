# Twitter Bot Detection

<p align="center">
    <a href="https://spark.apache.org/docs/latest/api/python/"><img alt="PySpark" src="https://img.shields.io/badge/-pyspark-important?logo=apachespark"></a>
    <a href="https://community.cloud.databricks.com/"><img alt="Databricks" src="https://img.shields.io/badge/-Databricks-yellow?logo=Databricks"></a>
    <a href="https://www.tweepy.org/"><img alt="Tweepy" src="https://img.shields.io/badge/-Tweepy-blue?logo=twitter"></a>
</p>

In this project, we are using a dataset from Kaggle and Botometer, containing in total 26K twitter user ids (twitter_human_bots_dataset_clean.csv). Thanks to the Tweepy Python library we are retrieving users' information, which we considered useful in order to determine the type of an account (retrieve_users.py). And then we perform some feature engineering on the data (One-Hot Encoding, Word2Vec)

We deal with numerical, categorical and also textual features. So, we train 3 types of models, one for the numerical/categorical features, another for the textual features and one more for the combination of the two types of features.

We also experimented with different classifiers (Logistic Regression, Decision Tree, Random Forests and Factorization Machines) and evaluated our results with different evaluation metrics (AUC, UPR).

Finally, with a simple Flask app, we can test our best model (Numerical Random Forest with AUC 0.948 and UPR 0.934).

PySpark, in combination with the Databricks platform, enables us to cope with this big data task efficiently.

Run
```
$ cd bot-website
$ docker build -t bot .
$ docker run -p 5000:5000 bot:latest
```
