from zipfile import ZipFile
import numpy as np
import gzip
import json
import csv
import os

# ROOT = 'Users/neal/Desktop/Challenge/'
ROOT = '/home/ec2-user'
TRAINING_JSON = os.path.join(ROOT, 'TrainingData.zip')
TRAINING_CATEGORIES = os.path.join(ROOT, 'TrainingDataCategories')
TOPICS = os.path.join(ROOT, 'topicDictionary.txt')


with open(TOPICS, 'r') as topics:
    categories = [r[0] for r in csv.reader(topics)]


with ZipFile(TRAINING_JSON) as zipfile:
    challenge_files = zipfile.namelist()
    for challenge_file in challenge_files:

        with zipfile.open(challenge_file, 'r') as lines:
            data = json.loads(lines.read())['TrainingData']

        for category in categories:
            print categories.index(category), category, challenge_file

            positive = []
            negative = []
            for e in data.values():
                if category in e['topics']:
                    positive.append(e)
                else:
                    negative.append(e)

            if len(positive) != 0:
                if len(negative) > len(positive):
                    if len(negative) > 5000:
                        negative = list(np.random.choice(negative, 5000, replace=False))

                result = os.path.join(TRAINING_CATEGORIES, category)
                if not os.path.exists(result):
                    os.makedirs(os.path.join(TRAINING_CATEGORIES, category))

                result = os.path.join(result, os.path.split(challenge_file)[1].replace('.json', '.json.gz'))
                with gzip.open(result, 'w') as out:
                    for e in positive + negative:
                        out.write(json.dumps(e) + '\n')
