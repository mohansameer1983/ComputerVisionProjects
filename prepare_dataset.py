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
    Extract Categories and Images Information into dictionary object
    """
    # Category Dictionary Format {'category id': 'category name'}
    categories_dict = {}

    for category in categories_info_arr:
        categories_dict[category['id']] = category['name']

    logging.info("Number of Categories: %s ; Categories: [%s]" % (nr_cats, categories_dict))

    # Image Dictionary Format {'image_id': 'file_name'}
    image_dict = {}

    for image in images_arr:
        image_dict[image['id']] = image['file_name']

    logging.info("Number of images: %s ; Images: [%s]" % (nr_images, image_dict))

    """
        Extract Images/Category/Annotations Information and construct final dataframe
        Final Images Dataframe Format Required:  {'annotation_id', 'image_id', 'category_id','file_name','category_name'}
    """

    # Extract required columns from annotations dataset
    final_info_df = pd.DataFrame(dataset['annotations'])[['id', 'image_id', 'category_id']]
    final_info_df.rename(columns={'id': 'annotation_id'}, inplace=True)

    logging.debug("Final_info_df Data: %s " % (final_info_df.head()))

    # Extract image and category relationship from annotations
    for index, row in final_info_df.iterrows():
        final_info_df.at[index, ['file_name', 'category_name']] = (
            image_dict.get(row['image_id']), categories_dict.get(row['category_id']))

    # Drop and rows containing NAN values and convert category id column to Integer
    final_info_df = final_info_df.dropna()
    final_info_df['category_id'] = final_info_df['category_id'].astype(int)
    logging.info("Final Dataframe Rows: %s " % (final_info_df.head(10)))
    logging.info(final_info_df.info())

    '''Filter dataframe where image ids/category ids are repeated in the row.
        e.g. if an image contains 3 same category plastic bottle, it will have 3 entries in annotations.json
        For our classification problem, we want to keep just 1 of the rows.
        If you are using this code for object detection or segmentation, than remove following line and include
        more annotation columns to consider bounded box details. 
    '''
    filtered_df = final_info_df.drop_duplicates(subset=['image_id', 'category_id'], keep='first')

    logging.info("Final filtered Dataframe Rows: %s " % (filtered_df.head(10)))
    logging.info(filtered_df.info())

    '''
    Converting dataframe to format: {image_name : [list of category labels]}. e.g.
    batch_1/000012.jpg              [Glass bottle, Other plastic wrapper]
    batch_1/000013.jpg                                     [Glass bottle]
    batch_1/000014.jpg  [Styrofoam piece, Drink can, Plastic film, Oth...
    batch_1/000015.jpg                       [Plastic film, Crisp packet]
    
    This is helpful to get process data into form which can be used for multi-label classification
    '''
    grouped_image_df = filtered_df.groupby('file_name')['category_name'].apply(list).reset_index(name='labels')
    logging.info("Final grouped Dataframe Rows: %s " % (grouped_image_df.head(25)))
    logging.info(grouped_image_df.info())

    return filtered_df, grouped_image_df, categories_dict


def create_folder(folder_data_path):
    """
    This function creates normal folder
    :param folder_data_path:
    :return:
    """
    dir_root = os.getcwd()
    dir_target = os.path.join(dir_root, folder_data_path)
    Path(dir_target).mkdir(parents=True, exist_ok=True)
    return


def create_split_folder(split_data_path, data_class_list):
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


def move_files_to_other_labelled_folder(source_path, destination_path, image_dataframe, split_into_category=False):
    """
    This function creates folders in the desired format
    :param source_path: Directory containing images without labelled folders
    :param image_dataframe: Dataframe with all information on images
        {'annotation_id', 'image_id', 'category_id','file_name','category_name'}
    :param destination_path: Destination with labelled subdirectories
    :param split_into_category: boolean value to indicate if files needed to be moved as per categories
    :return:
    """
    dir_root = os.getcwd()
    count = 0
    for index, row in image_dataframe.iterrows():
        # Retrieve category name of image
        category_name = row['category_name']
        # Retrieve file name of image
        file_name = str(row['file_name'])

        source_filepath = dir_root + os.sep + source_path + os.sep + file_name

        if not split_into_category:
            destination_filepath = dir_root + os.sep + destination_path + os.sep + \
                                   file_name.replace('/', '_')
        else:
            destination_filepath = dir_root + os.sep + destination_path + os.sep + category_name + os.sep + \
                                   file_name.replace('/', '_')

        logging.debug(source_filepath)
        logging.debug(destination_filepath)
        try:
            shutil.copyfile(source_filepath, destination_filepath)
            logging.debug(f'{count}. destination_filepath:{destination_filepath}')
            count = count + 1
        except Exception as e:
            logging.error(f'{sys.exc_info()[0]} occurred: {source_filepath}', exc_info=True)

    logging.info(f'Count:{count}')

    return


def save_dataframe_to_csv(dataframe, file_path):
    dir_root = os.getcwd()
    dataframe.to_csv(dir_root + os.sep + file_path, sep='\t')
    return


def main():
    dataset_path = 'data'
    anns_file_path = dataset_path + os.sep + 'annotations.json'
    train_images_path = dataset_path + os.sep + 'train'
    test_images_path = dataset_path + os.sep + 'test'
    split_data_path = dataset_path + os.sep + 'split'
    train_preprocessed_data_path = dataset_path + os.sep + 'train_preprocessed'
    test_preprocessed_data_path = dataset_path + os.sep + 'test_preprocessed'

    image_dataframe, grouped_image_dataframe, labels_dict = convert_annotation_json_to_dataframe(anns_file_path)
    # Optional step to save data to csv file
    save_dataframe_to_csv(image_dataframe, 'data/filtered_image_data.csv')
    save_dataframe_to_csv(grouped_image_dataframe, 'data/grouped_image_data.csv')
    create_folder(train_preprocessed_data_path)
    create_folder(test_preprocessed_data_path)
    #create_split_folder(split_data_path, labels_dict.values())
    move_files_to_other_labelled_folder(train_images_path, train_preprocessed_data_path, image_dataframe, False)
    move_files_to_other_labelled_folder(test_images_path, test_preprocessed_data_path, image_dataframe, False)

    return


if __name__ == "__main__":
    main()
