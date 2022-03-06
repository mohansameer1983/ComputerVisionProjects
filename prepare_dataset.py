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

import json


def read_annotation_json(file_path):
    # Read annotations from file
    with open(file_path, 'r') as file:
        dataset = json.loads(file.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    super_categories_name = []


    print("Number of categories: %s ; Categories: [%s]" % (nr_cats,categories))
    print('Number of annotations:', nr_annotations)
    print('Number of images:', nr_images)

    return


def create_folder(split_data_path, data_class, folder_type):
    """
    This function creates folders in the desired format
    :param split_data_path: where will the organized dataset is to be saved
    :param data_class: these are the categories inside dataset
    :param folder_type: train/validation/test
    :return: final path: dataset/split/train/categoryX
    """
    dir_root = os.getcwd()
    dir_target = os.path.join(dir_root, split_data_path, folder_type, data_class)
    Path(dir_target).mkdir(parents=True, exist_ok=True)
    return dir_target


def main():
    dataset_path = '.'
    anns_file_path = dataset_path + '/' + 'annotations.json'
    read_annotation_json(anns_file_path)


if __name__ == "__main__":
    main()
