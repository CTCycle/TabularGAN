import numpy as np
import os
from numpy.random import randn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from keras.layers import Normalization, IntegerLookup, StringLookup, CategoryEncoding, TextVectorization
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

   
        
# Class for preprocessing tabular data prior to GAN training 
#==============================================================================
#==============================================================================
#==============================================================================
class PreProcessingData:
    
            
        
    # preprocessing model for tabular data    
    #==============================================================================    
    def preprocess_dataset(self, dataframe):
        
        df = dataframe.copy()
        #----------------------------------------------------------------------        
        string_cols = [name for name, array in df.items() if df[name].dtype == 'object']
        cont_cols = [name for name, array in df.items() if df[name].dtype in ('float16', 'float32', 'float64')]
        all_int_cols = [name for name, array in df.items() if df[name].dtype in ('int32', 'int64')]
        binary_cols = [name for name, array in df.items() if df[name].dtype in ('int32', 'int64') and np.min(array) == 0 and np.max(array) == 1]         
        int_cols = [x for x in all_int_cols if x not in binary_cols]       
        #----------------------------------------------------------------------
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.scaler.fit(df[cont_cols])
        df[cont_cols] = self.scaler.transform(df[cont_cols])        
        #----------------------------------------------------------------------
        self.encoder_cache = {}
        for col in string_cols:
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            encoder_pair = {col : label_encoder}
            self.encoder_cache.update(encoder_pair)
        
        
            
        return df    

    
      
# Class that embeds all variables and functions for the simple GAN model
#==============================================================================
#==============================================================================
#==============================================================================
class TabularGAN:
    
    """ 
    GAN_framework()
    
    This class defines the model for the synthetic data generation using 
    Generative Adversarial Networks (GAN). Embedded functions are used to define
    the Deep Learning algorithms for both the generator and the discriminator.
    
    dataframe (pd.dataframe): target dataframe
                   
    """ 
    def __init__(self, device = 'default'):        
        self.n_epochs = 20000        
        self.batch_size = 128
        self.learning_rate = 0.000001       
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        self.available_devices = tf.config.list_physical_devices()
        print('----------------------------------------------------------------')
        print('The current devices are available: ')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('----------------------------------------------------------------')
        if device == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
            print('GPU is set as active device')
            print('----------------------------------------------------------------')
            print()
        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('----------------------------------------------------------------')
            print()
             
        
        
        
    # generate random points using latent dimensions. This functions is used to generate
    # inputs for the generator model
    #==========================================================================   
    def generate_latent_points(self, latent_dim, n_samples):
        
        """ 
        generate_latent_points(latent_dim, n_samples)
        
        Generates a series of random point in the latent space, which are used 
        as inputfor the generator model to produce complex dataseries. Returns the
        reshaped dataseries generated with random numbers.
         
        Keyword arguments:  
            
        latent_dim (int): number of dimensions of the latent space 
        n_samples (int):  number of samples that need to be generated 
        
        Returns: 
        
        x_input (list): list of values
                      
        """
        self.x_input = randn(latent_dim * n_samples)
        self.x_input = self.x_input.reshape(n_samples, latent_dim)
        
        return self.x_input
    
    # generate synthetic values using the latent points as inputs for the ML generator
    # model, and provides a series of labels with zeros to mark synthetic values as fake
    #==========================================================================
    def generate_fake_samples(self, generator, latent_dim, n_samples):
        
       """ 
       generate_fake_samples(generator, latent_dim, n_samples)
       
       Generates fake numbers based on the latent points generator previously defined.
       
        
       Keyword arguments:
           
       generator (tf model):  any generator instance (es. GAN generator model)
       latent_dim (int):      number of dimensions of the latent space (int)
       n_samples (int):       number of samples that need to be generated (int)
                     
       Returns: 
       
       X (numpy array): x-values 
       Y (numpy array): y-values 
       
       """       
       
       self.x_input = self.generate_latent_points(latent_dim, n_samples)
       self.x = generator.predict(self.x_input)       
       self.y = np.zeros((n_samples, 1))        
       
       return self.x, self.y
        
    # generate n real samples with class labels; We randomly select n samples 
    # from the real data array
    #==========================================================================
    def generate_real_samples(self, dataframe, n_samples):
        
        """
        This method samples a dataframe and returns the sample with its corresponding labels.

        Keyword arguments:  
            
        dataframe (pd.dataframe): The dataframe to be sampled
        n_samples (int): The number of samples to be taken

        Returns:
           
        X (pd.dataframe): the sampled dataframe
        y (numpy array):  an array of ones with shape (n_samples, 1)
       
        """
        self.X = dataframe.sample(n_samples)        
        self.y = np.ones((n_samples, 1))          
          
        
        return self.X, self.y

    # functional model as generator with Keras 
    #==========================================================================
    def generator_model(self, latent_dim, n_outputs):
        
        """ 
        generator_model(latent_dim, n_outputs)
        
        Defines the generator model based on Keras architecture (sequential models).
        The structure is define layer by layer and activation functions are set
        within each layer, according to the intput-output consistency. Number of neurons
        is calculated based on the actual outputs number. 
        
        Keyword arguments: 
            
        latent_dim (int): number of dimensions of the latent space 
        n_outputs (int):  number of outputs to be generated, corresponding to all columns
                          that are being modeled with the AI 
                    
        Returns:
            
        model (keras model): keras sequential model
                      
        """             
                
        
        input_layer = Input(shape = (latent_dim, ), name = 'Input_layer' )
        
        #----------------------------------------------------------------------
        input_batchnorm = BatchNormalization(momentum = 0.8, name = 'input_batchnorm')(input_layer)
        #----------------------------------------------------------------------
        input_dense_layer = Dense(512, activation = 'tanh', kernel_initializer ='he_uniform', 
                          name = 'dense_layer_1')(input_batchnorm)
        #----------------------------------------------------------------------
        dropout_1 = Dropout(0.4, name = 'dropout_1')(input_dense_layer)
        #----------------------------------------------------------------------
        dense_layer_2 = Dense(512, activation = 'LeakyReLU', name = 'dense_layer_2')(dropout_1)
        #----------------------------------------------------------------------
        batchnorm_2 = BatchNormalization(momentum = 0.8, name = 'batchnorm_2')(dense_layer_2)
        #----------------------------------------------------------------------
        dropout_2 = Dropout(0.3, name = 'dropout_2')(batchnorm_2)
        #----------------------------------------------------------------------
        dense_layer_3 = Dense(256, activation = 'relu', name = 'dense_layer_3')(dropout_2)
        #----------------------------------------------------------------------
        batchnorm_3 = BatchNormalization(momentum = 0.8, name = 'batchnorm_3')(dense_layer_3)
        #----------------------------------------------------------------------
        dropout_3 = Dropout(0.2, name = 'dropout_3')(batchnorm_3)
        #----------------------------------------------------------------------
        dense_layer_4 = Dense(256, activation = 'relu', name = 'dense_layer_4')(dropout_3)
        #----------------------------------------------------------------------
        batchnorm_4 = BatchNormalization(momentum = 0.8, name = 'batchnorm_4')(dense_layer_4)
        #----------------------------------------------------------------------
        dropout_4 = Dropout(0.2, name = 'dropout_4')(batchnorm_4)
        #----------------------------------------------------------------------
        dense_layer_5 = Dense(256, activation = 'relu', name = 'dense_layer_5')(dropout_4)
        #----------------------------------------------------------------------
        dense_layer_6 = Dense(128, activation = 'relu', name = 'g_dense_5')(dense_layer_5)
        #----------------------------------------------------------------------
        output = Dense(n_outputs, activation = 'relu')(dense_layer_6)       
        
        model = Model(inputs = input_layer, outputs = output, name = 'Generator')
        
        return model

    # functional model as discriminator with Keras 
    #==========================================================================
    def discriminator_model(self, n_inputs):
        
        """ 
        discriminator_model(n_inputs)
        
        Defines the discriminator model based on Keras architecture (sequential models).
        The structure is define layer by layer and activation functions are set
        within each layer, according to the intput-output consistency. Number of neurons
        is calculated based on the actual inputs number. 
        
        Keyword arguments:
            
        n_outputs (int):  number of inputs, corresponding to all columns that are being 
                          modeled with the AI        
        
        Returns: 
        
        model (keras model): keras functional model
                      
        """      
       
        input_layer = Input(shape = (n_inputs, ), name = 'Input')
        #----------------------------------------------------------------------
        input_batchnorm = BatchNormalization(momentum = 0.8, name = 'input_batchnorm')(input_layer)
        #----------------------------------------------------------------------
        input_dense_layer = Dense(512, activation = 'tanh', kernel_initializer ='he_uniform', 
                          name = 'dense_layer_1')(input_batchnorm)
        #----------------------------------------------------------------------
        dropout_1 = Dropout(0.4, name = 'dropout_1')(input_dense_layer)
        #----------------------------------------------------------------------
        dense_layer_2 = Dense(512, activation = 'LeakyReLU', name = 'dense_layer_2')(dropout_1)
        #----------------------------------------------------------------------
        batchnorm_2 = BatchNormalization(momentum = 0.8, name = 'batchnorm_2')(dense_layer_2)
        #----------------------------------------------------------------------
        dropout_2 = Dropout(0.3, name = 'dropout_2')(batchnorm_2)
        #----------------------------------------------------------------------
        dense_layer_3 = Dense(256, activation = 'relu', name = 'dense_layer_3')(dropout_2)
        #----------------------------------------------------------------------
        batchnorm_3 = BatchNormalization(momentum = 0.8, name = 'batchnorm_3')(dense_layer_3)
        #----------------------------------------------------------------------
        dropout_3 = Dropout(0.2, name = 'dropout_3')(batchnorm_3)
        #----------------------------------------------------------------------
        dense_layer_4 = Dense(256, activation = 'relu', name = 'dense_layer_4')(dropout_3)
        #----------------------------------------------------------------------
        dense_layer_5 = Dense(128, activation = 'relu', name = 'dense_layer_5')(dense_layer_4)
        #----------------------------------------------------------------------
        dense_layer_6 = Dense(64, activation = 'relu', name = 'dense_layer_6')(dense_layer_5)
        #----------------------------------------------------------------------        
        disc_output = Dense(1, activation = 'sigmoid')(dense_layer_6)
        
        
        model = Model(inputs = input_layer, outputs = disc_output,
                      name = 'Discriminator')
        
        opt = keras.optimizers.Adam(learning_rate = self.learning_rate)
        model.compile(loss = 'binary_crossentropy', optimizer = opt, 
                           metrics = ['accuracy'])
        
        return model
    
    # combined GAN model 
    #==========================================================================
    def GAN_model(self, latent_dim, n_outputs):
        
        """ 
        GAN_model(latent_dim, n_outputs)
        
        Defines the combined GAN model using predefined generator and discriminator models.
        The optimizer is selected within the model structure (default: Adam), and accuracy
        metrics and error calculation metrics can be defined as well. The discriminator
        is set as non trainable to avoid weight overriding during the iterative process
        
        Keyword arguments: 
            
        latent_dim (int): number of latent dimensions
        n_outputs (int):  number of outputs
                
        Returns: 
        
        model (keras model): keras functional model
                      
        """
        n_inputs = n_outputs       
        #----------------------------------------------------------------------
        generator = self.generator_model(latent_dim, n_outputs)
        discriminator = self.discriminator_model(n_inputs)        
        discriminator.trainable = False 
        #----------------------------------------------------------------------        
        input_gen = Input(shape = (latent_dim, ), name = 'Input_generator')        
        #----------------------------------------------------------------------
        output = generator(input_gen)
        #----------------------------------------------------------------------
        discriminator_output = discriminator(output)
                
        model = Model(inputs = input_gen, outputs = discriminator_output, name = 'GAN_model')

        opt = keras.optimizers.Adam(learning_rate = self.learning_rate)
        model.compile(loss = 'binary_crossentropy', optimizer = opt, 
                      metrics = ['accuracy'])
        
        return model
    
    # GAN loss and accuracy history plotting
    #==========================================================================
    def plot_history(self, dr_loss, df_loss, g_loss, dr_acc, df_acc, path):
        
        """ 
        plot_history(dr_loss, df_loss, g_loss, dr_acc, df_acc, path)
        
        Plots the accuracy and loss of the GAN model over the iterative process.
        Metrics are plotted referring to both the discriminator and generator 
        models, indicating the performance at generating realistic fake numbers 
        and discriminating real and fake numbers. The figure is saved into the
        target folder at the end of the process.
        
        Keyword arguments: 
            
        dr_loss (numpy array): loss function of the discriminator (real numbers)
        df_loss (numpy array): loss function of the discriminator (fake numbers)
        g_loss (numpy array):  loss function of the generator (fake numbers)
        dr_acc (numpy array):  accuracy of the discriminator (real numbers)
        df_acc (numpy array):  accuracy of the discriminator (fake numbers)
        path (str):            plot saving path 
        
        Returns:
        
        None
                              
        """
        plt.subplot(2, 1, 1)
        plt.plot(dr_loss, label = 'discriminator (real)')
        plt.plot(df_loss, label = 'discriminator (fake)')
        plt.plot(g_loss, label = 'generator (fake)')
        plt.legend(loc='best', fontsize = 8)
        plt.title('Loss plot GAN')
        plt.subplot(2, 1, 2)
        plt.plot(dr_acc, label = 'discriminator (real)')
        plt.plot(df_acc, label='discriminator (fake)')
        plt.legend(loc='best', fontsize = 8)
        plt.title('Accuracy plot GAN')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', format ='jpeg', dpi = 600)
        plt.show(block = False)
        plt.close()
    
    # training process for the GAN model
    #==========================================================================        
    def train_GAN(self, dataframe, g_model, d_model, gan_model, latent_dim, path_fig):
        
        """ 
        train_GAN(dataframe, g_model, d_model, gan_model, latent_dim, 
                  n_epochs, batch_size, n_eval, path_fig, path_mod)
        
        Defines the training procedure for the combined GAN model. Latent points are
        used to generate random noise, while the discriminator evaluates this output
        and compares it with the real dataset values. The generator uses the sequential
        model to improve the number generation in order to increase the accuracy 
        (fake numbers becomes more similar to real numbers). The trained model 
        is saved within the designated folder.
        
        Keyword arguments: 
            
        dataframe (pd.dataframe): dataframe to use as reference for the generation of synthetic values
        g_model (tf model):       instance of the generator model
        d_model (tf model):       instance of the discriminator model 
        gan_model (tf model):     instance of the combinaed GAN model  
        latent_dim (int):         number of dimensions of the latent space (int)  
        n_epochs (int):           number of iterations
        batch_size (int):         size of the random samples batch used to train/validate the generator
        n_eval (int):             coming soon...
        path_fig (str):           path of the plot folder
        path_mod (str):           path of the trained model folder   
                              
        """
        
        half_batch = int(self.batch_size/2) 
        self.dr_loss = []
        self.df_loss = [] 
        self.g_loss = []
        self.dr_acc = []
        self.df_acc = []
        for epoch in tqdm(range(self.n_epochs)):
            self.x_real, self.y_real = self.generate_real_samples(dataframe, half_batch) 
            self.x_fake, self.y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
            
            # update discriminator
            #------------------------------------------------------------------
            self.d_loss_real, self.d_real_acc = d_model.train_on_batch(self.x_real, self.y_real) 
            self.d_loss_fake, self.d_fake_acc = d_model.train_on_batch(self.x_fake, self.y_fake)
           
            # prepare points in latent space as input for the generator, and generates
            # inverted labels for the fake samples
            #------------------------------------------------------------------
            self.x_gan = self.generate_latent_points(latent_dim, self.batch_size)            
            self.y_gan = np.ones((self.batch_size, 1))
            
            # update the generator via discriminator error
            #------------------------------------------------------------------
            self.g_loss_gan_x, self.g_loss_gan_y = gan_model.train_on_batch(self.x_gan, self.y_gan) 
            self.dr_loss.append(self.d_loss_real)
            self.df_loss.append(self.d_loss_fake)
            self.g_loss.append(self.g_loss_gan_x)
            self.dr_acc.append(self.d_real_acc)
            self.df_acc.append(self.d_fake_acc)
            if epoch % 20 == 0:
                self.plot_history(self.dr_loss, self.df_loss, self.g_loss, 
                                  self.dr_acc, self.df_acc, path_fig)        
        
        
       
        
