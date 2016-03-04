from os import path

DATA_PATH = '/Users/neal/Desktop/kaggle-experiments/bnp/data'

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

FULL_TRAIN_FILE = path.join(DATA_PATH, TRAIN_FILE)
FULL_TEST_FILE = path.join(DATA_PATH, TEST_FILE)

ENCODED_TRAIN_FILE = path.join(DATA_PATH, 'encoded-' + TRAIN_FILE)
ENCODED_TEST_FILE = path.join(DATA_PATH, 'encoded-' + TEST_FILE)

TARGET_CLASS = 'target'
ROW_ID = 'ID'

# SUBMISSION_FILE = path.join(DATA_PATH, 'sample_submission.csv')

