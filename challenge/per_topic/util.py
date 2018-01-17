import gzip
import json
import csv
import os

ROOT = '/Users/neal/Desktop/Challenge/'
# ROOT = '/home/ec2-user'
TOPICS = os.path.join(ROOT, 'topicDictionary.txt')
TRAINING_CATEGORIES = os.path.join(ROOT, 'TrainingDataCategories')
TEST_DATA = os.path.join(ROOT, 'TestData.json.gz')


def get_categories():
    with open(TOPICS, 'r') as topics:
        categories = [r[0] for r in csv.reader(topics)]
    return categories


def load(source, category):
    xs = []
    ys = []
    print category, source
    for challenge_file in [os.path.join(source, f) for f in os.listdir(source) if 'json' in f]:
        with gzip.open(challenge_file, 'r') as lines:
            for example in (json.loads(l) for l in lines):
                xs.append(example['bodyText'])
                ys.append(1 if category in example['topics'] else 0)
    return xs, ys


def load_words(source, category):
    xs = []
    ys = []
    print source
    for challenge_file in [os.path.join(source, f) for f in os.listdir(source) if 'json' in f]:
        print category, challenge_file
        with gzip.open(challenge_file, 'r') as lines:
            for example in (json.loads(l) for l in lines):
                xs.append(example['bodyText'].lower().split(' '))
                ys.append(1 if category in example['topics'] else 0)
    return xs, ys


def test_entries():
    with gzip.open(TEST_DATA, 'r') as lines:
        for entry in (json.loads(l) for l in lines):
            yield entry['id'], entry['bodyText']