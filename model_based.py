from pyspark import SparkContext
import time
import math
import json
from pyspark.mllib.recommendation import ALS, Rating
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from collections import defaultdict



sc = SparkContext('local[*]', 'task2')

start = time.time()

neighborhoodNum = 4

# Load and parse the data
# trainingFilePath = "/Users/jamie/PycharmProjects/hw3/yelp_train.csv"
# validationFilePath = "/Users/jamie/PycharmProjects/hw3/yelp_val.csv"
trainingFilePath = "/Users/jamie/PycharmProjects/mengchieh_lee_competition/data/yelp_train.csv"
validationFilePath = "/Users/jamie/PycharmProjects/mengchieh_lee_competition/data/yelp_val.csv"
# userJsonFilePath = "/Users/jamie/PycharmProjects/mengchieh_lee_competition/user.json"
# busJsonFilePath = "/Users/jamie/PycharmProjects/mengchieh_lee_competition/business.json"
ratingRDD = sc.textFile(trainingFilePath)
validationRDD = sc.textFile(validationFilePath)
# userRatingRDD = sc.textFile(userJsonFilePath)
# busRatingRDD = sc.textFile(busJsonFilePath)
header = ratingRDD.first()


# # (user_id, business_id, stars)
# ratingRDD = ratingRDD.filter(lambda line: line != header).map(lambda line: line.split(","))\
#     .map(lambda t: (t[0], t[1], float(t[2])))
# validationRDD = validationRDD.filter(lambda line: line != header).map(lambda line: line.split(","))\
#     .map(lambda t: (t[0], t[1], float(t[2])))


reader = Reader(line_format='user item rating', sep=",", skip_lines=1)


folds_files = [(trainingFilePath, validationFilePath)]

data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()

algo = SVD()


predictionList = list()
for trainset, testset in pkf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    for uid, iid, true_r, est, _ in predictions:
        predictionList.append((uid, iid, est))

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


with open("model_based_output.csv", "w") as fp:
    fp.write("user_id, business_id, prediction\n")
    for x in predictionList:
        fp.write(str(x[0]) + "," + str(x[1]) + "," + str(x[2]) + "\n")

end = time.time()
print('Case3 Duration: ' + str(end - start))
