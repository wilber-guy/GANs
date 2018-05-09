#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:26:07 2018

@author: bud-guy
"""

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import os


class DCGAN():
    
    def __init__(self, img_rows=28, img_cols=28):
        # input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        
        optimizer = Adam(lr=0.002, beta_1 = 0.5)
        
        # build discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        
        # Build Generator
        self.generator = self.build_generator()
        
        # generator takes noise
        z = Input(shape=(100,))
        img = self.generator(z)
        
        
        # only train the generator
        self.discriminator.trainable = False
        
        valid = self.discriminator(img)
        
        # combined models
        self.combined = Model(z,valid, name='combined')
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    def build_generator(self):
        
        if os.path.isfile('./models/generator_model.h5'):
            model = load_model('./models/generator_model.h5')
        
        else:
            model = Sequential()
            
            model.add(Dense(128*7*7, activation='relu', input_shape=(self.latent_dim,)))
            model.add(Reshape((7, 7, 128)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(UpSampling2D())
            model.add(Conv2D(128, kernel_size=3, padding='same'))
            model.add(Activation('relu'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(UpSampling2D())
            model.add(Conv2D(64, kernel_size=3, padding='same'))
            model.add(Activation('relu'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
            model.add(Activation('tanh'))
            
            model.summary()

        
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)
    
    def build_discriminator(self):
        
        if os.path.isfile('./models/discriminator_model.h5'):
            model = load_model('./models/discriminator_model.h5')
        else:
            model = Sequential()
                
            model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
            model.add(ZeroPadding2D(padding=((0,1),(0,1))))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Conv2D(256,kernel_size=3, strides=1, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            
            model.add(Flatten())
            model.add(Dense(1,activation='sigmoid'))
            
        model.summary()

        
        img = Input(shape=self.img_shape)
        validity = model(img)
        
        return Model(img, validity)
    
    def train(self, epochs, batch_size=128, save_interval=50):
        
        # load dataset
        (X_train, _), (_, _) = mnist.load_data()
        
        # rescale to -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            
            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            
            # Sample random noise and gen half new batch of imgs
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)
            
            # Train on half real and half generated images
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))
            
            
            # Plot the progress
            print("{} [Discriminator Loss: {:.2f}, acc: {:.2f} [ Generator Loss: {:.2f}] ]".format(epoch, d_loss[0], 100*d_loss[1],g_loss))
            
            if epoch % save_interval == 0:
                	self.save_imgs(epoch)
                
                                
        self.save_models()

    def save_models(self):
        self.discriminator.save('./models/discriminator_model.h5' )
        self.generator.save('./models/generator_model.h5')
    
    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)
        
        # rescale images 0 to 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
                
            fig.savefig('images/mnist{}.png'.format(epoch))
            plt.close()
                
if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
