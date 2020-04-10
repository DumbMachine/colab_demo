'''
Will serve the data
'''
import cv2
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from defaults import LOCATIONS

from tqdm import tqdm
from keras.models import Sequential
from keras.applications import VGG16, ResNet50
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, GlobalAveragePooling2D


class ImageManager:

    def __init__(self, imagepaths, categories=None, sample_size=100, labels=None):
        '''
        path: path of the images
        labels: only used during the testing phase
        '''
        #  assinging the class variables
        self.paths = shuffle(imagepaths)
        self.sample_size = sample_size
        self.categories = ['street', 'mountain', 'sea', 'buildings', 'forest', 'glacier']
        # initializing the housekeeping variable
        self.current_image_index = 0
        self.train_image_index = dict([cat, 0] for cat in self.categories)
        self.dummy_image_index = dict([cat, self.train_image_index] for cat in self.categories)
        self.image_df = pd.DataFrame([[image, None] for image in self.paths], columns=['image','label'])


    def serveimage(self):
        '''
        Simple method to return image from the current index of things
        '''
        if self.current_image_index == len(self.paths):
            # return empty image or something
            raise Exception("All the images have now been served, nothing remains to be seen")
        
        current_image_path = self.paths[self.current_image_index]
        self.current_image_index += 1

        image = cv2.imread(current_image_path)
        return image

    # TESTONLY
    def skip_images(self, index):
        '''
        SKips the images by the index of number used
        '''
        # completing the labels uptill that point
        for i in range(self.current_image_index, index):
            label = self.paths[i].split("/")[-2]
            self.image_df.loc[i].label = label

        self.current_image_index += index

    def give_label(self, image_index, label):
        '''
        Will assing the label <label> to the image as the given index in the imagepahts array
        '''
        # self.image_df.loc[image_index] = [self.paths[image_index], label]
        self.image_df.loc[image_index].label = label
    
    def batch_from_images(self, category='sea'):
        '''
        Will make a trainer for all the image categories that have more images than the sample size
        '''

        if category is None:
            raise Exception(f"Category {category}, doesn't exist")
      
        train_batch, dumb = None, None

        # checking the images have the required ammount
        total_values = self.image_df[self.image_df.label == category].shape[0]
        print(f"searching for the batch: {total_values}")
        if total_values > self.sample_size and total_values > self.train_image_index[category]:
            train_batch = self.image_df[self.image_df.label == category].image.values[self.train_image_index[category]:self.train_image_index[category]+self.sample_size]
            print(f"Give the train batch for {category}: {self.train_image_index[category], self.train_image_index[category]+self.sample_size}")
            self.train_image_index[category] += self.sample_size

        if train_batch is not None:
            # creating the dummy batch now
            percent_from = int(len(train_batch) / (len(self.categories) - 1))
            
            dumb = []
            for cat in self.categories:
                if cat != category and train_batch is not None:
                    print(f"Give the train batch for {category}: {self.dummy_image_index[category][cat],self.dummy_image_index[category][cat]+percent_from}")
                    print(category, cat, self.dummy_image_index[category][cat])
                    dumb.extend(self.image_df[self.image_df.label == cat].image.values[self.dummy_image_index[category][cat]:self.dummy_image_index[category][cat]+percent_from])
                    self.dummy_image_index[category][cat] += percent_from

        if train_batch is None and dumb is None:
            print("Skipping, since not enough samples")

        return train_batch, dumb

    def train_on_category(self, category='sea'):
        a,b = self.batch_from_images(category=category)

        images = np.array([
            cv2.imread(path) for path in np.concatenate((a,b))
        ])
        labels = np.array(
            [1 for _ in range(len(a))]+[0 for _ in range(len(b))]
        )
        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.20)
        