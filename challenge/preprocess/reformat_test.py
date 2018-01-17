from zipfile import ZipFile
import gzip
import json
import os

# ROOT = 'Users/neal/Desktop/Challenge/'
ROOT = '/home/ec2-user'
TESTING_JSON = os.path.join(ROOT, 'TestData.zip')
TESTING_JSON_LINES = os.path.join(ROOT, 'TestData.json.gz')


with ZipFile(TESTING_JSON) as zipfile, gzip.open(TESTING_JSON_LINES, 'w') as out:
    challenge_file = zipfile.namelist()[0]
    print 'Load', challenge_file
    with zipfile.open(challenge_file, 'r') as lines:
        test_data = json.loads(lines.read())['TestData']
        for test_id in test_data:
            entry = test_data[test_id]
            entry['id'] = test_id
            out.write(json.dumps(entry) + '\n')
