from os import path

TARGET_CLASS = 'target'
ROW_ID = 'ID'

DATA_PATH = '/Users/neal/Desktop/kaggle-experiments/bnp/data'

SUBMISSION_FILE = path.join(DATA_PATH, 'sample_submission.csv')

FULL_TRAIN_FILE = path.join(DATA_PATH, 'train.csv')
FULL_TEST_FILE = path.join(DATA_PATH, 'test.csv')

ENCODED_TRAIN_FILE = path.join(DATA_PATH, 'encoded-train.csv')
ENCODED_TEST_FILE = path.join(DATA_PATH, 'encoded-test.csv')

SUBSET_TRAIN_FILE = path.join(DATA_PATH, 'subset-training.csv')
SUBSET_TEST_FILE = path.join(DATA_PATH, 'subset-testing.csv')

