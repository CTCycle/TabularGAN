import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model



    
# define the class for inspection of the input folder and generation of files list.
# The extension as argument allows identifying specific files (.csv, .xlsx, .pdf, etc)
# and making a list of those than can be called with the 'target_files' method
#==============================================================================
#==============================================================================
#==============================================================================
class DataSetFinder:
    
    """    
    A class to find, load and manipulate dataset files present in a given folder.
    If the folder is empty, prompts the user to add files in the designated path. 
     
    Methods:
        
    __init__(path):   initialize the class for the given directory path 
    dataset_loader(): allows for dataset selection
   
    """   
    def __init__(self, path):        
        self.path = path
        self.extensions = ('.csv', '.xlsx')
        os.chdir(path)
        self.all_files = os.listdir(path)
        self.target_files = [f for f in self.all_files if f.endswith(self.extensions)]   
        
    
# Select a trained model of .h5 extension and generate a dictionary of model names
# and actual loaded models
#==============================================================================
#==============================================================================
#==============================================================================
class ModelFinder:
    
    """
    model_selection(path)

    Loads the Keras model with .h5 extension and generates a dictionary where the keys
    are the model names and the values are the models themselves

    Keyword arguments:
        
    path (str): The path of the directory containing the target file
            
    Returns:
        
    models (dictionary): a dictionary containing the model names as keys and the
                         loaded model as values
    
    """ 
    def __init__(self, path):
        os.chdir(path)
        self.path = path
        self.models = os.listdir(path)
        
    
    #==========================================================================
    def model_loader(self):    
        
        
        
        index_list = [idx + 1 for idx, name in enumerate(self.models)] 
        if len(self.models) > 0:
            print('The following models are available for data generation')
            print()       
            for idx, name in enumerate(self.models):
                print(idx + 1, ' - ', name)
            
            print()
            while True:
                try:
                    model_idx = int(input('Select the model from the list: '))            
                except:
                    continue        
                while model_idx not in index_list:
                    try:
                        model_idx = int(input('Input is not valid Try again: '))
                        print()
                    except:
                        continue
                break
        else:
            print('No model has been found! Select a GAN model and launch training')
            pass
        
        
        model_name = self.models[model_idx - 1]    
        GAN_model = load_model(os.path.join(model_name, 'pretrained_GAN'))
        preprocess_model = load_model(os.path.join(model_name, 'preprocess_model'))
        
        return preprocess_model, GAN_model
        
        
        
   
# define the class for the operation with the input dataset to be fed to the GAN network
#==============================================================================
#==============================================================================
#==============================================================================
class DataSetOperations:
    
    """ 
    DataSetOperations(dataframe)
    
    Initialize the dataframe prior to machine learning augmentation using the GAN network
        
    Keyword arguments: 
        
    dataframe (pd.dataframe): target dataframe
            
    Returns:
        
    None
        
    """    
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    # Split the dataframe into sub-datasets based on their variables type     
    #==========================================================================
    def columns_by_type(self):
        
        """        
        columns_by_type()
    
        Split the initialized dataframe into partitions that include only columns 
        of homogeneous variable types (continuous, categoricals and binomials). 
    
        Keyword arguments:
            
        None
        
        Returns:
            
        df_continuous (pd.DataFrame):   partitioned dataframe containing only continuous variables
        df_categoricals (pd.DataFrame): partitioned dataframe containing only categrocal variables
        df_binomials (pd.DataFrame):    partitioned dataframe containing only binomial variables
        
        """
        
        all_categoricals = [col for col in self.dataframe.columns if self.dataframe[col].dtype == 'int32' or self.dataframe[col].dtype == 'int64']
        continuous = [col for col in self.dataframe.columns if self.dataframe[col].dtype == 'float32' or self.dataframe[col].dtype == 'float64']
        binomials = [col for col in all_categoricals if self.dataframe[col].min() == 0 and self.dataframe[col].max() == 1]
        true_categoricals = [x for x in all_categoricals if x not in binomials] 

        df_continuous = self.dataframe[continuous]
        df_categoricals = self.dataframe[true_categoricals]
        df_binomials = self.dataframe[binomials]

        self.num_categoricals = df_categoricals.shape[1]
        self.num_continuous = df_continuous.shape[1]
        self.num_binomials = df_binomials.shape[1]

        return df_continuous, df_categoricals, df_binomials 


# define class for correlations calculations
#==============================================================================
#==============================================================================
#==============================================================================
class MultiCorrelator:
    
    """ 
    MultiCorrelator(dataframe)
    
    Calculates the correlation matrix of a given dataframe using specific methods.
    The internal functions retrieves correlations based on Pearson, Spearman and Kendall
    methods. This class is also used to plot the correlation heatmap and filter correlations
    from the original matrix based on given thresholds. Returns the correlation matrix
    
    Keyword arguments: 
        
    dataframe (pd.dataframe): target dataframe
    
    Returns:
        
    df_corr (pd.dataframe): correlation matrix in dataframe form
                
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    # Spearman correlation calculation
    #==========================================================================
    def Spearman_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'spearman').round(decimals)
        return self.df_corr
    
    # Kendall correlation calculation
    #==========================================================================    
    def Kendall_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'kendall').round(decimals)
        return self.df_corr
    
    # Pearson correlation calculation
    #==========================================================================    
    def Pearson_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'pearson').round(decimals)
        return self.df_corr
    
    # plotting correlation heatmap using seaborn package
    #==========================================================================
    def corr_heatmap(self, matrix, path, dpi, name):
        
        """ 
        corr_heatmap(matrix, path, dpi, name)
        
        Plot the correlation heatmap using the seaborn package. The plot is saved 
        in .jpeg format in the folder that is specified through the path argument. 
        Output quality can be tuned with the dpi argument.
        
        Keyword arguments:    
            
        matrix (pd.dataframe): target correlation matrix
        path (str):            picture save path for the .jpeg file
        dpi (int):             value to set picture quality when saved (int)
        name (str):            name to be added in title and filename
        
        Returns:
            
        None
            
        """
        cmap = 'YlGnBu'
        sns.heatmap(matrix, square = True, annot = False, 
                    mask = False, cmap = cmap, yticklabels = False, 
                    xticklabels = False)
        plt.title('{}_correlation_heatmap'.format(name))
        plt.tight_layout()
        plot_loc = os.path.join(path, '{}_correlation_heatmap.jpeg'.format(name))
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
        plt.show(block = False)
        plt.close()
        
    # plotting correlation heatmap of two dataframes
    #==========================================================================
    def double_corr_heatmap(self, matrix_real, matrix_fake, path, dpi):        
        
        """ 
        double_corr_heatmap(matrix_real, matrix_fake, path, dpi)
        
        Plot the correlation heatmap of two dataframes using the seaborn package. 
        The plot is saved in .jpeg format in the folder specified through the path argument. 
        Output quality can be tuned with the dpi argument.
        
        Keyword arguments:
            
        matrix_real (pd.dataframe): real data correlation matrix
        matrix_fake (pd.dataframe): fake data correlation matrix
        path (str):                 picture save path for the .jpeg file
        dpi (int):                  value to set picture quality when saved (int)
        
        Returns:
            
        None
        
        """ 
        plt.subplot(2, 1, 1)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(matrix_real, square=True, annot=False, mask = False, 
                    cmap=cmap, yticklabels=False, xticklabels=False)
        plt.title('Real data')
        plt.subplot(2, 1, 2)
        sns.heatmap(matrix_fake, square=True, annot=False, mask = False, 
                    cmap=cmap, yticklabels=False, xticklabels=False)
        plt.title('Synthetic data')
        plt.tight_layout()
        plot_loc = os.path.join(path, 'Correlation_heatmap.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
        plt.show()
        plt.close()
     
    # filtering of correlation pairs based on threshold value. Strong, weak and null
    # pairs are isolated and embedded into output lists
    #==========================================================================    
    def corr_filter(self, matrix, threshold): 
        
        """
        corr_filter(matrix, path, dpi)
        
        Generates filtered lists of correlation pairs, based on the given threshold.
        Weak correlations are those below the threshold, strong correlations are those
        above the value and zero correlations identifies all those correlation
        with coefficient equal to zero. Returns the strong, weak and zero pairs lists
        respectively.
        
        Keyword arguments:    
        matrix (pd.dataframe): target correlation matrix
        threshold (float):     threshold value to filter correlations coefficients
        
        Returns:
            
        strong_pairs (list): filtered strong pairs
        weak_pairs (list):   filtered weak pairs
        zero_pairs (list):   filtered zero pairs
                       
        """        
        self.corr_pairs = matrix.unstack()
        self.sorted_pairs = self.corr_pairs.sort_values(kind="quicksort")
        self.strong_pairs = self.sorted_pairs[(self.sorted_pairs) >= threshold]
        self.strong_pairs = self.strong_pairs.reset_index(level = [0,1])
        mask = (self.strong_pairs.level_0 != self.strong_pairs.level_1) 
        self.strong_pairs = self.strong_pairs[mask]
        
        self.weak_pairs = self.sorted_pairs[(self.sorted_pairs) >= -threshold]
        self.weak_pairs = self.weak_pairs.reset_index(level = [0,1])
        mask = (self.weak_pairs.level_0 != self.weak_pairs.level_1) 
        self.weak_pairs = self.weak_pairs[mask]
        
        self.zero_pairs = self.sorted_pairs[(self.sorted_pairs) == 0]
        self.zero_pairs = self.zero_pairs.reset_index(level = [0,1])
        mask = (self.zero_pairs.level_0 != self.zero_pairs.level_1) 
        self.zero_pairs_P = self.zero_pairs[mask]
        
        return self.strong_pairs, self.weak_pairs, self.zero_pairs
