from zipfile import ZipFile
import csv
import json
import os

ROOT = '/Users/neal/Desktop/Challenge/'

TOPICS = os.path.join(ROOT, 'topicDictionaryList.txt')
TESTING_JSON = os.path.join(ROOT, 'TestData.zip')
SAMPLE = os.path.join(ROOT, 'sampleSubmission.csv')
BASELINE = os.path.join(ROOT, 'baseline.csv')

with open(TOPICS, 'r') as topics:
    rows = csv.reader(topics)
    categories = {''.join(r): [c for c in r if c not in ['and', 'of']] for r in rows}

for c in categories:
    print c, categories[c]

with ZipFile(TESTING_JSON) as zipfile:
    challenge_file = zipfile.namelist()[0]
    print 'Load', challenge_file
    with zipfile.open(challenge_file, 'r') as lines:
        data = json.loads(lines.read())['TestData']

with open(SAMPLE, 'r') as lines, open(BASELINE, 'w') as out:
    rows = csv.reader(lines)
    header = rows.next()
    id_column = header.index('id')

    result = csv.writer(out)
    result.writerow(header)

    for row in rows:
        print row[id_column]
        predictions = [row[id_column]]
        text = data[row[id_column]]['bodyText'].lower()
        for i in range(1, len(header)):
            category = categories[header[i]]
            predictions.append(1 if all(e in text for e in category) else 0)
        result.writerow(predictions)
