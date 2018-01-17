from sklearn import metrics
import numpy as np
import gzip
import csv
import json
import os

"""
For each category:
    1. Loads the data
    2. Splits it by training/test
    3. Vectorizes and Tf-Idf preprocessing
    4. Writes the result
"""

TOPICS = '/Users/neal/Desktop/Challenge/topicDictionaryList.txt'
TRAINING_CATEGORIES = '/Users/neal/Desktop/Challenge/TrainingDataCategories'

with open(TOPICS, 'r') as topics:
    rows = csv.reader(topics)
    categories = [r for r in rows]

f1s = []
limit = 5000

for category in categories:
    full_category = ''.join(category)
    category = [c for c in category if c not in ['and', 'of']]
    source = os.path.join(TRAINING_CATEGORIES, full_category)
    if os.path.exists(source):
        challenge_files = [os.path.join(source, f) for f in os.listdir(source) if '.json.gz' in f]
        y_true = []
        y_pred = []
        total = 0
        for challenge_file in challenge_files:
            with gzip.open(challenge_file, 'r') as lines:
                for entry in (json.loads(l) for l in lines):
                    text = entry['bodyText'].lower()
                    y_pred.append(1 if all(e in text for e in category) else 0)
                    y_true.append(1 if full_category in entry['topics'] else 0)
                    total += 1
                    if total == limit:
                        break
            if total == limit:
                break
        f1 = metrics.f1_score(y_true, y_pred)
        print category, total, f1
        f1s.append(f1)

print np.mean(f1s)
