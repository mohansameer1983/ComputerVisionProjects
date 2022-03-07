"""
This program organizes data in classes (labels) within Train, Validation and Test folders
using annotation file provided to classify images
Original Folder:
dataset/

Desired Output Folder:
dataset/
    split/
        train/
                category1/
                category2/
                ....
        validation/
                category1/
                category2/
                ....
        test/
                category1/
                category2/
                ....
"""

import os
from pathlib import Path
import shutil
import sys
import json

import pandas as pd
import numpy as np

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def convert_annotation_json_to_dataframe(annotation_json_file_path):
    """
        This function takes annotation json file path and convert all relevant information to dataframe
        :param annotation_json_file_path: file is stored here
        :return: image_info_df: dataframe containing relevant information on images
        :return: super_categories_dict: contains list of all the categories found in input json file
    """

    # Read annotations from file
    with open(annotation_json_file_path, 'r') as file:
        dataset = json.loads(file.read())

    categories_info_arr = dataset['categories']
    annotations_arr = dataset['annotations']
    images_arr = dataset['images']
    nr_cats = len(categories_info_arr)
    nr_annotations = len(annotations_arr)
    nr_images = len(images_arr)

    logging.debug("Number of categories: %s ; Categories: [%s]" % (nr_cats, categories_info_arr))
    logging.debug("Number of Images: %s ; Images: [%s]" % (nr_images, images_arr))
    logging.debug('Number of annotations:', nr_annotations)
    logging.debug('Number of images:', nr_images)

    """
    Extract Super Categories Information
    """
    # Super Category Dictionary Format {'category id': 'category name'}
    super_categories_dict = {}

    for super_categories in categories_info_arr:
        super_categories_dict[super_categories['id']] = super_categories['supercategory']

    logging.info("Number of categories: %s ; Categories: [%s]" % (nr_cats, super_categories_dict))

    """
        Extract Images Information and its relationship with categories
    """
    # Images Dataframe Format {'image id','file names','category id','category name'}
    annotations_dict = {}
    for annotation in annotations_arr:
        annotations_dict[annotation['image_id']] = annotation['category_id']

    logging.info("Number of Annotations: %s ; Annotations Data: [%s]" % (nr_annotations, annotations_dict))

    # Annotations Dataframe Format {'image id',''category id'}
    image_info_df = pd.DataFrame(dataset['images'])[['id', 'file_name']]

    # Extract image and category relationship from annotations
    for index, row in image_info_df.iterrows():
        category_id = annotations_dict.get(row['id'])
        image_info_df.at[index, ['supercategory_name', 'supercategory_id']] = (
            super_categories_dict.get(category_id), category_id)

    # Drop and rows containing NAN values and convert category id column to Integer
    image_info_df = image_info_df.dropna()
    image_info_df['supercategory_id'] = image_info_df['supercategory_id'].astype(int)
    logging.info(image_info_df.info())
    logging.info(image_info_df.head())

    return image_info_df, super_categories_dict


def create_folder(split_data_path, data_class_list):
    """
    This function creates folders in the desired format
    :param data_class_list: these are the categories inside dataset
    :param split_data_path: where will the organized dataset is to be saved
    :return:
    """
    dir_root = os.getcwd()
    for label in data_class_list:
        dir_target = os.path.join(dir_root, split_data_path, label)
        Path(dir_target).mkdir(parents=True, exist_ok=True)
    return


def move_files_to_other_labelled_folder(source_path, destination_path, image_dataframe):
    """
    This function creates folders in the desired format
    :param source_path: Directory containing images without labelled folders
    :param image_dataframe: Dataframe with all information on images
    :param destination_path: Destination with labelled subdirectories
    :return:
    """
    dir_root = os.getcwd()
    count = 0
    for index, row in image_dataframe.iterrows():
        # Retrieve category name of image
        category_name = row['supercategory_name']
        # Retrieve file name of image
        file_name = str(row['file_name'])

        source_filepath = dir_root + os.sep + source_path + os.sep + file_name
        destination_filepath = dir_root + os.sep + destination_path + os.sep + category_name + os.sep + \
                               file_name.replace('/', '_')

        logging.debug(source_filepath)
        logging.debug(destination_filepath)
        try:
            shutil.copyfile(source_filepath, destination_filepath)
            logging.info(f'{count}. destination_filepath:{destination_filepath}')
            count = count + 1
        except Exception as e:
            logging.error(f'{sys.exc_info()[0]} occurred: {source_filepath}', exc_info=True)

    logging.info(f'Count:{count}')
    return


def main():
    dataset_path = 'data'
    anns_file_path = dataset_path + os.sep + 'annotations.json'
    split_data_path = dataset_path + os.sep + 'split'
    source_images_path = dataset_path + os.sep + 'orig'

    image_dataframe, labels_dict = convert_annotation_json_to_dataframe(anns_file_path)
    create_folder(split_data_path, labels_dict.values())
    move_files_to_other_labelled_folder(source_images_path, split_data_path, image_dataframe)


if __name__ == "__main__":
    main()
