import os
import sys
import numpy as np
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
        preprocess_model, GAN_model = model_selection.model_loader()   
        
        

    # [GENERATE VIASUAL SCHEME OF THE MODEL]
    #==========================================================================
    if event == '-SHOWCASE-':               
        model_name = values['-MODELS-']
        model_picture_name = '{}_model.png'.format(model_name)
        model_plot_path = os.path.join(model_path, model_picture_name) 
        model_selection = ModelFinder(model_path)
        
        ML_model.summary()
        plot_model(ML_model, to_file = model_plot_path, show_shapes = True, 
                   show_layer_names = True, show_layer_activations = True, 
                   expand_nested = True, rankdir = 'TB', dpi = 400)           
        

    # [PREPROCESS DATA USING ADEQUATE PIPELINE]
    #==========================================================================
    if event == '-PREPROCESS-':
        pass


# # [LOAD PRETRAINED MODELS]
# #==============================================================================
# # Loading trained module and generating synthetic values upon applying variables
# # type mask based on generated numbers
# #==============================================================================
# model_selection = ModelFinder(model_path)
# preprocess_model, GAN_model = model_selection.model_loader() 

# # [PREPROCESS DATA]
# #==============================================================================
# # checking for model to call proper predict function and split dataset based on
# # variable types 
# #==============================================================================
# data_preprocessing = PreProcessingData()
# preprocessed_data = data_preprocessing.preprocess_tabular_dataset(df)
# normalizer = data_preprocessing.scaler


# # [GENERATE SYNTHETIC DATA]
# #==============================================================================
# # checking for model to call proper predict function and split dataset based on
# # variable types 
# #==============================================================================
# while True:
#     try:
#         synth_values = int(input('Select how many synthetic values you want to generate: '))
#         print()
#     except:
#         continue    
#     while synth_values <= 0:
#         try:
#             synth_values = int(input('Only positive values are allowed: '))
#             print()
#         except:
#             continue
#     break

# latent_dim = df.shape[1] + 1
# latent_points = TabularGAN.generate_latent_points(latent_dim, synth_values)
# generated_values = GAN_model.predict(latent_points)

# df_synthetic = pd.DataFrame(data = generated_values, columns = df.columns)       
    

  

# # [SAVE SYNTHETIC DATA]
# #==============================================================================
# print()
# print('-------------------- GAN synthetic dataframe -----------------------')
# print()
# print(df_synthetic)
# print()
# print('Saving dataframe into excel file...') 
# file_loc = os.path.join(save_path, 'Synthetic_GAN.xlsx')
# writer = pd.ExcelWriter(file_loc, engine = 'xlsxwriter')
# df_synthetic.to_excel(writer, sheet_name = 'Synthetic_data')
# writer.save()
# print()
# print('File has been saved!') 


# # [VALIDATE SYNTHETIC DATA]
# #==============================================================================
# # Plotting the histograms of real vs synthetic data to evaluate distribution of
# # generated data againts the empirically sampled values. Then, plotting the 
# # cumulative distribtuion function of real vs synthetic data, while estimating the 
# # distribution similarity by means of the Kolmogorov–Smirnov test. A list of P values 
# # and notes related to the KS test are generated and saved within a check file using pandas
# #==============================================================================
# print('----------------------------------------------------------------------')
# print('DATA VALIDATION: synthetic data is compared with original real data\n'
#       'to check integrity of data upon generation')
# print()
# print('STEP 1 - Histograms of both real and synthetic data')
# print()
# validation = ValidationGAN(df, df_synthetic)
# validation.hist_comparison(30, save_path) 

# print('STEP 2 - CDF analysis and Kolmogorov-Smirnoff test')
# print()
# validation.data_check(df, df_synthetic, save_path)
# pv_list = validation.pv_list
# real_list = validation.real_list
# fake_list = validation.fake_list

# desc_list = []        
# for f in pv_list:
#     if f <= 0.05:
#         desc = 'Generated distribution is not equal'
#     else:
#         desc = 'Generated distribution is equal'
#     desc_list.append(desc)

# df_dist_data = {'Variable': df.columns, 
#                 'P_value':pv_list, 
#                 'Notes':desc_list}

# file_loc = os.path.join(save_path, 'KS_test.xlsx')
# df_dist = pd.DataFrame(df_dist_data).T
# writer = pd.ExcelWriter(file_loc, engine = 'xlsxwriter')
# df_dist.to_excel(writer, sheet_name='KS_test')
# writer.save()    

# # [VALIDATE DATA CORRELATIONS] 
# #==============================================================================       
# # Plotting the correlation heatmap of the synthetic and real data distributions, and
# # saving both the correlation matrix and the filtered list of pairs with strong, weak and
# # zero-type correlations 
# #==============================================================================
# print('STEP 3 - Correlation heatmap')
# print()
# regressor_real = MultiCorrelator(df)
# regressor_fake = MultiCorrelator(df_synthetic)
# df_corr_real = regressor_real.Spearman_corr(df, 2)
# df_corr_fake = regressor_fake.Spearman_corr(df_synthetic, 2)
# regressor_real.double_corr_heatmap(df_corr_real, df_corr_fake, save_path, 600)

# # script end 
# #==============================================================================
# if __name__ == '__main__':
#         pass       
