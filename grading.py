import json
import os, sys
import pandas as pd
import math
import numpy as np


def format_checking(input_file):
    try:
        your_prediction_result = pd.read_csv(input_file, skiprows=1, names=['user_id', 'business_id', 'prediction'])
        return your_prediction_result
    except Exception as e:
        print('File format is wrong')
        exit(1)


def grading(prediction_result, ground_truth):

    try:
        evaluation = prediction_result.merge(ground_truth, on=['user_id', 'business_id'])

        if len(evaluation) == 0:
            print('No overlapping pairs; ')
            return -1
        elif len(evaluation) != len(ground_truth):
            print('Less predictions; ')
            return -1
        else:
            prediction = np.asarray(evaluation['prediction'], dtype=float)
            stars = np.asarray(evaluation['stars'], dtype=float)
            delta = (prediction - stars) ** 2
            return math.sqrt(np.mean(delta))

    except Exception as e:
        print(e)
        exit(1)


if __name__ == '__main__':

    setting_file = sys.argv[1]
    json_data = open(setting_file).read()

    # json_data = open('data/settings2.json').read()
    settings = json.loads(json_data)

    ground_truth_path = settings['ground_truth_path']
    result_path = settings['result_path']

    ground_truth = pd.read_csv(ground_truth_path, skiprows=1, names=['user_id', 'business_id', 'stars'])

    for _, _, files in os.walk(result_path):
        for file in files:
            if file.startswith('.') or file.startswith('_'):
                continue

            else:
                file_path = os.path.join(result_path, file)
                prediction_result = format_checking(file_path)
                rmse = grading(prediction_result, ground_truth)
                if rmse > 0:
                    print('{} RMSE = {}'.format(file, rmse))


