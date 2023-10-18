# [IMPORT PACKAGES AND SETTING WARNINGS]
#==============================================================================
import os
import sys
import numpy as np
from datetime import date
import pickle
import threading
from os.path import dirname
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.utils.vis_utils import plot_model
import PySimpleGUI as sg
import warnings
warnings.simplefilter(action='ignore', category = DeprecationWarning)
warnings.simplefilter(action='ignore', category = FutureWarning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.components.data_classes import DataSetFinder, ModelFinder, MultiCorrelator, DataSetOperations
from modules.components.training_classes import PreProcessingData, TabularGAN
from modules.components.validation_classes import ValidationGAN

# [DEFINE PATHS]
#==============================================================================
if getattr(sys, 'frozen', False):    
    model_path = os.path.join(os.path.dirname(dirname(sys.executable)), 'pretrained models')   
else:   
    model_path = os.path.join(os.path.dirname(dirname(os.path.abspath(__file__))), 'pretrained models')       

if not os.path.exists(model_path):
    os.mkdir(model_path)

# [WINDOW THEME AND OPTIONS]
#==============================================================================
sg.theme('LightGrey1')
sg.set_options(font = ('Arial', 11), element_padding = (6, 6))

# [LAYOUT OF THE PREPROCESSING FRAME]
#==============================================================================
pp_text = sg.Text('Timeseries has not been preprocessed', font = ('Arial', 12), key = '-PPTEXT-')
norm_checkbox = sg.Checkbox('Normalize data', key = '-NORMCHECK-')
savedata_checkbox = sg.Checkbox('Save preprocessed data', key = '-SAVECHECK-')
preprocess_button = sg.Button('Preprocess data', key = '-PREPROCESS-', expand_x= True, disabled=True)
pp_frame = sg.Frame('Preprocessing parameters', layout = [[norm_checkbox],
                                                          [savedata_checkbox],
                                                          [pp_text],
                                                          [preprocess_button]], expand_x = True)

# [LAYOUT OF THE TRAINING PARAMETERS FRAME]
#==============================================================================
dev_input_text = sg.Text('Training device', expand_x = True, pad = (3,3), font = ('Arial', 9))
devices_input = sg.DropDown(['CPU', 'GPU'], size = (8,1), key = '-DEVICE-', expand_x = True,enable_events=True)
model_input_text = sg.Text('AI model', expand_x = True, pad = (3,3), font = ('Arial', 9))
model_input = sg.DropDown(['StandardGAN'], size = (8,1), key = '-MODELS-', expand_x = True, enable_events=True)
lr_input_text = sg.Text('Learning rate', expand_x = True, pad = (3,3), font = ('Arial', 9))
learning_rate_input = sg.Input(key = '-LR-', size = (8,1), expand_x = True, enable_events=True)
epochs_input_text = sg.Text('Epochs', expand_x = True, pad = (3,3), font = ('Arial', 9))
epochs_input = sg.Input(key = '-EPOCHS-', size = (8,1), expand_x = True, enable_events=True)
bs_input_text = sg.Text('Batch size', expand_x = True, pad = (3,3), font = ('Arial', 9))
batch_size_input = sg.Input(key = '-BS-', size = (8,1), expand_x = True, enable_events=True)
pretrain_button = sg.Button('Pretrain model', key = '-PRETRAIN-', expand_x= True, disabled=True)
modelshow_button = sg.Button('Show model scheme', key = '-SHOWCASE-', expand_x= True, disabled=True)
pt_frame = sg.Frame('Pretraining parameters', layout = [[dev_input_text, devices_input],
                                                        [model_input_text, model_input],
                                                        [lr_input_text, learning_rate_input],
                                                        [bs_input_text, batch_size_input],
                                                        [epochs_input_text, epochs_input],                                                        
                                                        [modelshow_button],
                                                        [pretrain_button]], expand_x = True)
                                   
# [LAYOUT OF OUTPUT AND CANVAS]
#==============================================================================
output = sg.Output(size = (100, 10), key = '-OUTPUT-', expand_x = True)
canvas_object = sg.Canvas(key='-CANVAS-', size=(500, 500), expand_x=True)

# [LAYOUT OF THE WINDOW]
#==============================================================================
left_column = sg.Column(layout = [[pp_frame], [pt_frame]])
right_column = sg.Column(layout = [[canvas_object]])
training_layout = [[left_column, sg.VSeparator(), right_column],
                   [sg.HSeparator()],
                   [output]]                           

# [WINDOW LOOP]
#==============================================================================
training_window = sg.Window('Pretraining using machine learning', training_layout, 
                            grab_anywhere = True, resizable=True, finalize = True)

while True:
    event, values = training_window.read()

    if event == sg.WIN_CLOSED:
        break            
       
          

    if event == '-MODELS-':
        training_window['-SHOWCASE-'].update(disabled=False)
               
        

    # [GENERATE VIASUAL SCHEME OF THE MODEL]
    #==========================================================================
    if event == '-SHOWCASE-':               
        model_name = values['-MODELS-']
        model_picture_name = '{}_model.png'.format(model_name)
        model_plot_path = os.path.join(model_path, model_picture_name) 
          
        GAN_model = GAN_framework.GAN_model(latent_dim, num_of_cols)  
        GAN_model.summary()

        gan_scheme_path = os.path.join(pretrained_path, 'GAN_scheme.png')
        plot_model(GAN_model, to_file = gan_scheme_path, show_shapes = True, 
                   show_layer_names = True, show_layer_activations = True, 
                   rankdir = 'TB', expand_nested = True, dpi = 400)
                
        

    # [PREPROCESS DATA USING ADEQUATE PIPELINE]
    #==========================================================================
    if event == '-PREPROCESS-':
        model_name = 'TabGAN_{}'.format(str(date.today()))
        gan_model_path = os.path.join(model_path, model_name)
        if not os.path.exists(gan_model_path):
            os.mkdir(gan_model_path) 
            
        pp_data_path = os.path.join(data_path, 'preprocessed datasets')
        if not os.path.exists(pp_data_path):
            os.mkdir(pp_data_path)

        data_preprocessing = PreProcessingData()
        preprocessed_data = data_preprocessing.preprocess_dataset(df)
        normalizer = data_preprocessing.scaler
        encoders = data_preprocessing.encoder_cache        
        
        today_date = str(date.today())
        file_loc = os.path.join(pp_data_path, 'PP_{0}_{1}.csv'.format(filename, today_date))
        preprocessed_data.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')

    # [PREPROCESS DATA USING ADEQUATE PIPELINE]
    #==========================================================================
    if event == '-PRETRAIN-':
        GAN_framework = TabularGAN(device = 'GPU')
        print('------------------------- Generator model specifics -------------------------')
        print()
        generator = GAN_framework.generator_model(latent_dim, num_of_cols)
        generator.summary()
        print('------------------------- Discriminator model specifics ----------------------')
        print()
        discriminator = GAN_framework.discriminator_model(num_of_cols)
        discriminator.summary()
        gan_model_path = os.path.join(model_path, model_name)
        if not os.path.exists(gan_model_path):
            os.mkdir(gan_model_path)
            
        training_plot_path = os.path.join(gan_model_path, 'Loss_and_accuracy_GAN.jpeg')
        GAN_framework.train_GAN(preprocessed_data, generator, discriminator, GAN_model, 
                                latent_dim, training_plot_path) 

        GAN_model.save(pretrained_path)


training_window.close() 







            

