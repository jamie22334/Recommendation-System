from pyspark import SparkContext
import time
import math
import json

sc = SparkContext('local[*]', 'task2')

# Load and parse the data
trainingFilePath = "/Users/jamie/PycharmProjects/mengchieh_lee_competition/yelp_train.csv"
validationFilePath = "/Users/jamie/PycharmProjects/mengchieh_lee_competition/yelp_val.csv"
busJsonFilePath = "/Users/jamie/PycharmProjects/mengchieh_lee_competition/business.json"
ratingRDD = sc.textFile(trainingFilePath)
validationRDD = sc.textFile(validationFilePath)
busRatingRDD = sc.textFile(busJsonFilePath)
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


# (business_id, [categories])
categoryRDD = busRatingRDD.map(lambda line: json.loads(line))\
    .map(lambda arr: (arr["business_id"], str(arr["categories"])))\
    .join(preUserRDD.map(lambda t: (t, -1))).map(lambda t: (t[0], t[1][0]))
# print(categoryRDD.top(3))

categoryAll = categoryRDD.flatMap(lambda t: t[1].split(",")).map(lambda t: t.strip()).distinct()
print("category count: " + str(categoryAll.count()))

# catMap = dict()
# catRDD = categoryRDD.map(lambda t: t[1].split(","))
# for arr in catRDD.collect():
#     for x in arr:
#         x = x.strip()
#         if x not in catMap:
#             catMap[x] = 0
#         catMap[x] = catMap[x] + 1
#
# catList = list()
# for key in catMap:
#     catList.append((catMap[key], key))
#
# catList.sort(reverse=True)
# for x in catList:
#     print(x)

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

for key in itemProfileMap:
    print(str(key) + "," + str(itemProfileMap[key]))
    break
