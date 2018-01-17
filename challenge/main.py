import csv
import numpy as np

from per_topic.per_topic_experiment import results_by_category

with open('best-preprocessor-per-category.csv', 'w') as out:
    rows = csv.writer(out)

    scores = []
    for result in results_by_category():
        scores.append(result[1])
        print result, np.mean(scores)
        rows.writerow(result)
