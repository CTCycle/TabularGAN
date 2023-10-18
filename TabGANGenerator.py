# [IMPORT PACKAGES AND SETTING WARNINGS]
#==============================================================================
import os
import sys
import numpy as np
import pandas as pd
import PySimpleGUI as sg
import warnings
warnings.simplefilter(action='ignore', category = DeprecationWarning)
warnings.simplefilter(action='ignore', category = FutureWarning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.components.data_classes import DataSetFinder
import modules.global_variables as GlobVar

# [DEFAULT FOLDER PATHS]
#==============================================================================
if getattr(sys, 'frozen', False):    
    initial_folder = os.path.dirname(sys.executable)
else:    
    initial_folder = os.path.dirname(os.path.realpath(__file__))

# [WINDOW THEME AND OPTIONS]
#==============================================================================
sg.theme('LightGrey1')
sg.set_options(font = ('Arial', 11), element_padding = (6,6))


# [LAYOUT OF THE FILE SELECTION FRAME]
#==============================================================================
list_of_files = GlobVar.list_of_files
input_text = sg.Text('Input folder', font = ('Arial', 12), size = (10,1))
dd_text = sg.Text('List of files', font = ('Arial', 12), size = (10,1))
input_path = sg.Input(enable_events=True, key= '-LOADPATH-', size = (70,1))
input_browser = sg.FolderBrowse(initial_folder = initial_folder, key = '-INBROWSER-')
dropdown = sg.DropDown(list_of_files, size = (20, 1), key = '-DROPDOWN-', expand_x = True, enable_events=True)
path_frame = sg.Frame('Select folder path', layout = [[input_text, input_path, input_browser], [dd_text, dropdown]],
                                                       expand_x=True)

# [LAYOUT OF OPERATIONS FRAME]
#==============================================================================
train_button = sg.Button('Generative Adversarial Network (GAN) pretraining', expand_x=True, key = '-TRAINOPS-', disabled=True)
generator_button = sg.Button('Generate synthetic values using pretrained GAN', expand_x=True, key = '-GENOPS-', disabled=True)
validation_button = sg.Button('Validate synthetic data', expand_x=True, key = '-VALOPS-', disabled=True)
operations_frame = sg.Frame('Select desired operation', layout = [[train_button], 
                                                                  [generator_button],
                                                                  [validation_button]], expand_x=True)

# [LAYOUT OF THE WINDOW]
#==============================================================================
main_text = sg.Text('Placeholder text', font = ('Arial', 12), size = (60,1))
main_layout = [[main_text],
               [path_frame],
               [sg.HSeparator()],
               [operations_frame],
               [sg.HSeparator()]]    
                          

# [WINDOW LOOP]
#==============================================================================
main_window = sg.Window('Simple table generator V1.0', main_layout, 
                        grab_anywhere = True, resizable = True, finalize = True)
while True:
    event, values = main_window.read()

    if event == sg.WIN_CLOSED:
        break   

    # [SELECT FILES USING DROPDOWN MENU]
    #==========================================================================
    if event == '-DROPDOWN-':
        target_file = values['-DROPDOWN-'] 
        folder_path = values['-LOADPATH-']     
        filepath = os.path.join(folder_path, target_file)                
        df = pd.read_csv(filepath, sep= ';', encoding='utf-8')
        GlobVar.dataframe_name = target_file.split('.')[0]   
        GlobVar.dataframe, GlobVar.clean_dataframe = df, df.copy()                
        main_window['-TRAINOPS-'].update(disabled = False)           
            

    if event == '-LOADPATH-':
        path = values['-LOADPATH-']
        dataset_inspector = DataSetFinder(path)
        list_of_files = dataset_inspector.target_files
        GlobVar.list_of_files = list_of_files
        main_window['-DROPDOWN-'].update(values = list_of_files) 

    # [LAUNCH DATA DISTRIBUTION]
    #==========================================================================
    if event == '-TRAINOPS-':
        import modules.TabGAN_training
        del sys.modules['modules.TabGAN_training']

    # [LAUNCH DATA DISTRIBUTION]
    #==========================================================================
    if event == '-GENOPS-':
        import modules.TabGAN_generator
        del sys.modules['modules.TabGAN_generator']


main_window.close() 
