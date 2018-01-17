"""
Inspiration:
https://www.kaggle.com/inversion/processing-bson-files
"""

import io
import os
import sys
import bson
import csv
from skimage.data import imread
from skimage.io import imsave


def get_categories(depth, source='category_names.csv'):
    with open(source, 'r') as lines:
        rows = csv.reader(lines)
        rows.next()  # category_id,category_level1,category_level2,category_level3
        categories = {int(row[0]): row[depth] for row in rows}
    return categories


def split_images(source, depth):
    categories = get_categories(depth)
    with open(source, 'rb') as data:
        for entry in bson.decode_file_iter(data):
            product_id = entry['_id']
            category = categories[int(entry['category_id'])]
            target = os.path.join('data', 'categories' + str(depth), category)

            if not os.path.exists(target):
                print target
                os.makedirs(target)

            for e, pic in enumerate(entry['imgs']):
                picture = imread(io.BytesIO(pic['picture']))
                picture_file = os.path.join(target, str(product_id) + '_' + str(e) + '.jpg')
                imsave(picture_file, picture)


if __name__ == '__main__':
    bson_file = sys.argv[1]
    category_depth = int(sys.argv[2])
    split_images(bson_file, category_depth)
