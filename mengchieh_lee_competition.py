from pyspark import SparkContext
import time
import math
from random import shuffle
from pyspark.mllib.recommendation import ALS, Rating
import json
from sklearn.metrics.pairwise import cosine_similarity
import sys
from surprise import SVD
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from sklearn.linear_model import LinearRegression
import numpy as np

sc = SparkContext('local[*]', 'task2')

neighborhoodNum = 2

similarityThreshold = 0.2
minHashNum = 180
outputFileName = sys.argv[3]

# b bands, r rows/band (60, 3)
b = 60
r = int(minHashNum / b)
print("b: " + str(b) + " r: " + str(r))

start = time.time()

# Load and parse the data
trainingFilePath = sys.argv[1] + "yelp_train.csv"
validationFilePath = sys.argv[2]
userJsonFilePath = sys.argv[1] + "user.json"
busJsonFilePath = sys.argv[1] + "business.json"

ratingRDD = sc.textFile(trainingFilePath)
validationRDD = sc.textFile(validationFilePath)
userRatingRDD = sc.textFile(userJsonFilePath)
busItemRDD = sc.textFile(busJsonFilePath)
header = ratingRDD.first()


# (user_id, business_id, stars)
ratingRDD = ratingRDD.filter(lambda line: line != header).map(lambda line: line.split(","))\
    .map(lambda t: (t[0], t[1], float(t[2])))
validationRDD = validationRDD.filter(lambda line: line != header).map(lambda line: line.split(","))\
    .map(lambda t: (t[0], t[1], float(t[2])))


# set(user_id)
preUserRDD = ratingRDD.map(lambda t: t[0]).distinct()
preUserRDD2 = validationRDD.map(lambda t: t[0]).distinct()
preUserRDD = preUserRDD.union(preUserRDD2).distinct()
preUserList = preUserRDD.collect()
preUserMap = dict()
for i in range(len(preUserList)):
    preUserMap[preUserList[i]] = i


# set(business_id)
preBusinessRDD = ratingRDD.map(lambda t: t[1]).distinct()
preBusinessRDD2 = validationRDD.map(lambda t: t[1]).distinct()
preBusinessRDD = preBusinessRDD.union(preBusinessRDD2).distinct()
preBusinessList = preBusinessRDD.collect()
preBusinessMap = dict()
for i in range(len(preBusinessList)):
    preBusinessMap[preBusinessList[i]] = i


# (user_id, average_stars)
userRatingRDD = userRatingRDD.map(lambda line: json.loads(line))\
    .map(lambda arr: (arr["user_id"], arr["average_stars"]))
userRatingRDD = userRatingRDD.join(preUserRDD.map(lambda t: (t, -1))).map(lambda t: (t[0], t[1][0]))
print("user avg count: " + str(userRatingRDD.count()))
userAvgMap = dict(userRatingRDD.collect())

# (business_id, stars)
busRatingRDD = busItemRDD.map(lambda line: json.loads(line)).map(lambda arr: (arr["business_id"], arr["stars"]))
busRatingRDD = busRatingRDD.join(preBusinessRDD.map(lambda t: (t, -1))).map(lambda t: (t[0], t[1][0]))
print("bus avg count: " + str(busRatingRDD.count()))
avgMap = dict(busRatingRDD.collect())


# (business_id, [categories])
categoryRDD = busItemRDD.map(lambda line: json.loads(line))\
    .map(lambda arr: (arr["business_id"], str(arr["categories"])))\
    # .join(preUserRDD.map(lambda t: (t, -1))).map(lambda t: (t[0], t[1][0]))
print("bus item count: " + str(categoryRDD.count()))

# categoryAll = categoryRDD.flatMap(lambda t: t[1].split(",")).map(lambda t: t.strip()).distinct()
# print("category count: " + str(categoryAll.count()))

categoryArr = ["Restaurants", "Beauty & Spas", "Hair Salons", "Food", "Skin Care",
               "Nightlife", "Coffee & Tea", "Nail Salons", "Bars", "American (New)"]
featureCount = 10


def build_item_profile(item):
    businessId = item[0]
    selfCategory = item[1]
    profileArray = [0] * featureCount

    for i in range(len(categoryArr)):
        c = categoryArr[i]
        if c in selfCategory:
            profileArray[i] = 1

    return businessId, profileArray


categoryRDD = categoryRDD.map(build_item_profile)
itemProfileMap = dict(categoryRDD.collect())


def user_id_to_index(line):
    userIndex = int(preUserMap[line[0]])
    businessIndex = int(preBusinessMap[line[1]])
    stars = float(line[2])

    return userIndex, businessIndex, stars


def user_id_to_index_without_rating(line):
    userIndex = int(preUserMap[line[0]])
    businessIndex = int(preBusinessMap[line[1]])

    return userIndex, businessIndex


def user_index_back_to_id(line):
    userId = preUserList[line[0]]
    businessId = preBusinessList[line[1]]
    stars = max(min(line[2], 5.0), 1.0)

    return userId, businessId, stars


preRatingRDD = ratingRDD.map(user_id_to_index).map(lambda t: Rating(t[0], t[1], t[2]))

# # Build the recommendation model using Alternating Least Squares
# rank = 3
# numIterations = 10
# numLambda = 0.009
# model = ALS.train(preRatingRDD, rank, numIterations, numLambda)
#
# # Evaluate the model on validation data
# validationData = validationRDD.map(user_id_to_index_without_rating)
# predictions = model.predictAll(validationData).map(user_index_back_to_id)
#
# print("model prediction total: " + str(predictions.count()))
#
# ratingRDD = ratingRDD.union(predictions)


ratingMap = dict()
for x in ratingRDD.collect():
    ratingMap[(x[0], x[1])] = x[2]
# print("total data: " + str(len(ratingMap)))


# add avg business stars
# avgBusList = list()
# userRatingRDD = userRatingRDD.filter(lambda t: int(preUserMap[t[0]]) % 30 == 0)
# busRatingRDD = busRatingRDD.filter(lambda t: int(preBusinessMap[t[0]]) % 40 == 0)
# print("add user avg count: " + str(userRatingRDD.count()))
# print("add bus avg count: " + str(busRatingRDD.count()))


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


# ((user_id, business_id), rating)
predictionSVD = sc.parallelize(predictionList).map(lambda t: ((t[0], t[1]), t[2]))


ratingMap = dict()
for x in ratingRDD.collect():
    ratingMap[(x[0], x[1])] = x[2]
print("total data: " + str(len(ratingMap)))


# set(user)
userOnlyRDD = ratingRDD.map(lambda t: t[0]).distinct()
userIndexList = userOnlyRDD.collect()
userNum = len(userIndexList)
userIndexMap = dict()
for i in range(userNum):
    userIndexMap[userIndexList[i]] = i

# (business, set(user))
businessRDD = ratingRDD.map(lambda t: (t[1], t[0])).groupByKey().mapValues(set)
businessMap = dict(businessRDD.collect())


# generate random matrix (#minHashNum, list(random))
randomMatrix = list()
for i in range(minHashNum):
    arr = [x for x in range(userNum)]
    shuffle(arr)
    randomMatrix.append(arr)


def generate_hash_value(userId):
    userIndex = userIndexMap[userId]
    transposeList = list()

    for i in range(len(randomMatrix)):
        transposeList.append(randomMatrix[i][userIndex])

    return userId, transposeList


def generate_min_hash(list1, list2):
    finalList = []

    for i in range(len(list1)):
        finalList.append(min(list1[i], list2[i]))

    return finalList


def generate_signature_list(minHashPair):
    resultList = []

    businessId = minHashPair[0]
    hashList = minHashPair[1]
    for band in range(b):
        subList = hashList[band * r: (band+1) * r]
        tupleKey = (band, tuple(subList))
        resultList.append((tupleKey, businessId))

    return resultList


def generate_business_candidate(bandTuple):
    candidateList = []

    businessList = bandTuple[1]
    for b1 in businessList:
        for b2 in businessList:
            if b1 < b2:
                candidateList.append((b1, b2))
    return candidateList


def count_jaccard_similarity(pair):

    set1 = businessMap[pair[0]]
    set2 = businessMap[pair[1]]
    union = len(set1.union(set2))
    intersection = len(set1.intersection(set2))

    similarity = intersection / union
    # print(str(intersection) + " / " + str(union) + " = " + str(similarity))
    return pair[0], pair[1], similarity


# (user_id, list(hashValue))
hashValueRDD = userOnlyRDD.map(generate_hash_value)

# (business_id, list(hashValue))
joinRDD = ratingRDD.map(lambda t: (t[0], t[1])).join(hashValueRDD).map(lambda t: (t[1][0], t[1][1]))

# (business_id, list(each_min_hash))
minHashRDD = joinRDD.reduceByKey(generate_min_hash)

# do LSH
# (bandId, signature) -> list(business)
bandRDD = minHashRDD.flatMap(generate_signature_list).groupByKey().mapValues(list).filter(lambda t: len(t[1]) >= 2)
candidates = bandRDD.flatMap(generate_business_candidate).distinct()
print("candidate count: " + str(candidates.count()))


# count similarity
resultRDD = candidates.map(count_jaccard_similarity).filter(lambda t: t[2] >= similarityThreshold)
print("result count: " + str(resultRDD.count()))


# Item-based CF
correlationMap = dict()
busCandidateSet = set()
noCorated = 0
for pair in resultRDD.collect():
    business1 = pair[0]
    business2 = pair[1]
    busCandidateSet.add(business1)
    busCandidateSet.add(business2)

    set1 = set(businessMap[business1])
    set2 = set(businessMap[business2])
    coRatedUser = set1.intersection(set2)
    count = len(coRatedUser)
    correlation = 0

    if count > 0:
        sum1 = 0
        sum2 = 0

        for user in coRatedUser:
            sum1 += ratingMap[(user, business1)]
            sum2 += ratingMap[(user, business2)]

        avg1 = sum1 / count
        avg2 = sum2 / count

        numerator = 0
        denominator1 = 0
        denominator2 = 0
        for user in coRatedUser:
            numerator += (ratingMap[(user, business1)] - avg1) * (ratingMap[(user, business2)] - avg2)
            denominator1 += pow(ratingMap[(user, business1)] - avg1, 2)
            denominator2 += pow(ratingMap[(user, business2)] - avg2, 2)

        denominator1 = math.sqrt(denominator1)
        denominator2 = math.sqrt(denominator2)
        if denominator1 * denominator2 != 0:
            correlation = numerator / (denominator1 * denominator2)

        # item profile
        feature1 = [itemProfileMap[business1]]
        feature2 = [itemProfileMap[business2]]
        cos = cosine_similarity(feature1, feature2)
        # print("cos similarity: " + str(cos))

        correlation = correlation * 0.7 + float(cos[0][0]) * 0.3

        if correlation != 0:
            if business1 not in correlationMap:
                correlationMap[business1] = list()
            if business2 not in correlationMap:
                correlationMap[business2] = list()
            correlationMap[business1].append((business2, correlation))
            correlationMap[business2].append((business1, correlation))

    else:
        noCorated += 1
print("bus candidate set: " + str(len(busCandidateSet)))
print("no corated pair: " + str(noCorated))


def take_second(elem):
    return -elem[1]


greaterCounter = 0
smallerCounter = 0

for businessKey in correlationMap:
    correlationMap[businessKey].sort(key=take_second)
    if len(correlationMap[businessKey]) >= neighborhoodNum:
        greaterCounter += 1
    else:
        smallerCounter += 1
print("greater counter: " + str(greaterCounter))
print("smaller counter: " + str(smallerCounter))

useAvg = 0
useCF = 0

for pair in validationRDD.collect():
    userId = pair[0]
    businessId = pair[1]

    if businessId not in correlationMap:
        useAvg += 1
    else:
        useCF += 1
print("use avg: " + str(useAvg))
print("use CF: " + str(useCF))


def generate_prediction(pair):
    userId = pair[0]
    businessId = pair[1]

    if businessId not in correlationMap:
        result = 0
        if businessId in avgMap:
            result = avgMap[businessId]
        elif userId in userAvgMap:
            result = userAvgMap[userId]
        return userId, businessId, result
    else:
        numerator = 0
        denominator = 0
        connectedBusiness = correlationMap[businessId]
        coRatedCount = 0
        prediction = 0

        if len(connectedBusiness) >= neighborhoodNum:
            for v in connectedBusiness:
                if coRatedCount == neighborhoodNum or v[1] < 0:
                    break

                # even if 2 items are similar, they might not be co-rated by the same user
                if (userId, v[0]) in ratingMap:
                    weight = v[1]
                    numerator += ratingMap[(userId, v[0])] * weight
                    denominator += abs(weight)
                    coRatedCount += 1

        if denominator != 0:
            prediction = numerator / denominator
            prediction = max(min(prediction, 5), 1)
        if prediction == 0 or coRatedCount < neighborhoodNum:
            prediction = avgMap[businessId]
        return userId, businessId, prediction


# Evaluate the model on training data
# ((user_id, business_id), rating)
predictionsCF = validationRDD.map(generate_prediction).map(lambda t: ((t[0], t[1]), t[2]))

# ((user_id, business_id), (CF, SVD, val))
totalPrediction = predictionsCF.join(predictionSVD).join(validationRDD.map(lambda t: ((t[0], t[1]), t[2])))\
    .map(lambda t: (t[0], (t[1][0][0], t[1][0][1], t[1][1])))
# print(totalPrediction.top(3))

regXList = list()
regYList = list()

for p in totalPrediction.collect():
    regXList.append((p[1][0], p[1][1]))
    regYList.append(p[1][2])

# print("xList: " + str(regXList))
# print("yList: " + str(regYList))
regX = np.array(regXList)
regY = np.array(regYList)
reg = LinearRegression().fit(regX, regY)

# ((user_id, business_id), (val, (CF, SVD, val))) -> (CF, SVD)
regPrediction = validationRDD.map(lambda t: ((t[0], t[1]), t[2])).join(totalPrediction)\
    .map(lambda t: (t[0], (t[1][1][0], t[1][1][1])))
# print(regPrediction.top(3))


def do_linear_regression(line):
    ids = line[0]
    cf = line[1][0]
    svd = line[1][1]

    # print("np array: " + str(cf) + "," + str(svd))
    final = reg.predict(np.array([[cf, svd]]))[0]
    # print("svd predict: " + str(final))

    return ids, final


regPrediction = regPrediction.map(do_linear_regression)
# print(regPrediction.top(3))

ratesAndPredictions = validationRDD.map(lambda t: ((t[0], t[1]), t[2])).join(regPrediction)
MSE = ratesAndPredictions.map(lambda r: (r[1][0] - r[1][1])**2).mean()
RMSE = math.sqrt(MSE)
print("Root Mean Squared Error = " + str(RMSE))

with open(outputFileName, "w") as fp:
    fp.write("user_id, business_id, prediction\n")
    for x in regPrediction.collect():
        fp.write(str(x[0][0]) + "," + str(x[0][1]) + "," + str(x[1]) + "\n")


end = time.time()
print('Duration: ' + str(end - start))
