from collections import defaultdict
import csv
import os

ROOT = '/Users/neal/Desktop/Challenge/'
SAMPLE = os.path.join(ROOT, 'sampleSubmission.csv')

PREDICTIONS = os.path.join(ROOT, 'TopicPreprocessedPredictions.csv')
BASELINE = os.path.join(ROOT, 'Submissions', 'baseline.csv')

RESULT = os.path.join(ROOT, 'Submissions', 'TopicPreprocessedSubmissionNoBaseline.csv')


predictions = defaultdict(dict)
with open(os.path.join(PREDICTIONS), 'r') as lines:
    rows = csv.reader(lines)
    for row in rows:
        # afghanistan,TestData_02085,1
        predictions[row[0]][row[1]] = row[2]

# with open(BASELINE, 'r') as lines:
#     rows = csv.reader(lines)
#     header = rows.next()
#     for row in rows:
#         test_entry = row[0]
#         for i in range(1, len(header)):
#             topic = header[i]
#             if test_entry not in predictions[topic]:
#                 predictions[topic][test_entry] = row[i]


with open(SAMPLE, 'r') as lines, open(RESULT, 'w') as out:
    samples = csv.reader(lines)
    result = csv.writer(out)

    header = samples.next()
    result.writerow(header)

    for sample in samples:
        test_entry = sample[0]
        print test_entry
        row = [test_entry]
        for i in range(1, len(header)):
            topic = header[i]
            row.append(predictions[topic].get(test_entry,0))
            # row.append(predictions[topic].get(test_entry, 0))
        result.writerow(row)
