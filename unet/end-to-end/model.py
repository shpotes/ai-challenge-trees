import math
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from datetime import datetime
from architecture import vanilla_unet
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, Tensorboard

class UNET:
    def __init__(self, config):
        self.config = config
        self.metadata = pd.read_csv(self.config['model']['metadata_path'])
        self.model = self.build_model()


    def build_model(self):
        return eval('%s(%s)' % (self.config['model']['architecture'],
                           self.config['model']['input_size']))


    def preprocess(self, img, mask=None, padding_mode='CONSTANT'):
        insize = int(self.config['model']['input_size'])
        ousize = int(self.config['model']['output_size'])
        img = tf.image.resize(img, (ousize, ousize))
        img /= 255.0
        mask = tf.image.resize(tf.expand_dims(mask, axis=-1), (ousize, ousize))
        
        if self.config['model']['architecture'] == 'vanilla_unet':
            pad = [[(insize - ousize) // 2] * 2] * 2 + [[0, 0]]
            img = tf.pad(img, pad, padding_mode)

        return img, mask


    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)


    def save_weights(self, weight_path):
        self.model.save_weights(weight_path)


    def _create_data(self, test=False):
        BATCH_SIZE = self.config['train']['batch_size']
        NUM_EPOCHS = self.config['train']['num_epochs']
        TEST_SIZE = get_folder_size(self.metadata, 'test')
        VAL_SIZE = get_folder_size(self.metadata, 'val')
        metadata = self.metadata

        train_source = build_source_from_metadata(
            metadata,
            self.config['model']['data_path'],
            'train'
        )
        
        print(np.load(self.config['model']['mask_path']).shape)
        train_data = make_dataset(
            train_source,
            mask = np.load(self.config['model']['mask_path']),
            training=True,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            preprocess=lambda x, y: self.preprocess(x, y)
        )

        if test:
            test_source = build_source_from_metadata(
                self.metadata,
                self.config['model']['data_path'],
                'test'
            )
            test_data = make_dataset(
                test_source,
                mask = np.load(self.config['model']['mask_path']),
                training=False,
                batch_size=TEST_SIZE,
                num_epochs=NUM_EPOCHS,
                num_parallel_calls=-1,
                preprocess=lambda x, y: self.preprocess(x, y)
            )

            return train_data, test_data

        val_source = build_source_from_metadata(
            metadata,
            self.config['model']['data_path'],
            'val'
        )
        val_data = make_dataset(
            val_source,
            mask = np.load(self.config['model']['mask_path']),
            training=False,
            batch_size=VAL_SIZE,
            num_epochs=NUM_EPOCHS,
            preprocess=lambda x, y: self.preprocess(x, y)
        )

        return train_data, val_data


    def jaccard_loss(self, y_true, y_pred, smooth=100):
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
        sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth


    def train(self):
        LR = self.config['train']['learning_rate']
        BATCH_SIZE = self.config['train']['batch_size']
        NUM_EPOCHS = self.config['train']['num_epochs']
        METRICS = self.config['train']['metrics']
        _CALLBACKS = self.config['train']['callbacks']

        self.model.compile(loss=self.jaccard_loss,
                           optimizer=optimizers.Adam(LR),
                           metrics=METRICS)
        train_data, val_data = self._create_data()

        CALLBACKS = [] if not _CALLBACKS \
            else [EarlyStopping(patience=10),
                  Tensorboard(log_dir='%s/log_%s_%s' % 
                              (self.config['train']['callbacks'],
                               self.config['model']['architecture'],
                               datetime.now().strftime("%Y%m%d-%H%M%S"))),
                  TerminateOnNaN(),
                  ReduceLROnPlateau(),
                  ModelCheckpoint('%s/chpts/w_%s.{epoch:02d}_%s.h5' % 
                                  (self.config['train']['callbacks'],
                                   self.config['model']['architecture'],
                                   datetime.now().strftime("%Y%m%d-%H%M%S")))]
                  
        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=NUM_EPOCHS,
            steps_per_epoch=BATCH_SIZE,
            validation_steps=4,
            callbacks=CALLBACKS)

    def evaluate(self):
        # TODO
        pass

    def predict(self, img):
        train_data, test_data = self._create_data(test=True)
