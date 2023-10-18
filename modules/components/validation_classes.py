import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp


# define class for trained model validation and data comparison
#============================================================================== 
#==============================================================================
#==============================================================================
class ValidationGAN:
    
    """ 
    ValidationGAN(dataframe1, dataframe2)
    
    Defines the functions to compare real and synthetic dataframes and evalutate the
    overall quality of the generation process. 
    
    Keyword arguments: 
        
    dataframe1 (pd.dataframe):  dataframe of real numbers (original dataframe)
    dataframe2 (pd.dataframe):  dataframe of fake numbers (synthetic dataframe)
    
    """
    def __init__(self, dataframe1, dataframe2):
        self.dataframe1 = dataframe1
        self.dataframe2 = dataframe2
    
    # comparison of histograms (distributions) by superimposing plots
    #========================================================================== 
    def hist_comparison(self, bins, path):
        
        """ 
        hist_comparison(bins, path)
        
        Plots the histograms of both the real and fake dataframe, column by column,
        using a mild transparency to superimpose them in a clear fashion. Standard
        deviation and mean differences are printed into a text box inside the plot 
        
        Keyword arguments:    
        bins (int):                 number of histogram bins (int)
        path (str):                 figures (.jpeg) folder path
        
        Returns:
        
        None
        
        """
        for (r, f) in tqdm(zip(self.dataframe1.columns, self.dataframe2.columns)):
            r_arr = self.dataframe1[r].values
            f_arr = self.dataframe2[f].values
            r_mu = r_arr.mean()
            f_mu = f_arr.mean()
            r_sigma = r_arr.std()
            f_sigma = f_arr.std()
            std_check = (abs(r_sigma - f_sigma)/r_sigma)*100
            mean_check = (abs(r_mu - f_mu)/r_mu)*100
            std_check = round(std_check, 2)
            mean_check = round(mean_check, 2)
            text = '''STD diff = {0}%
                      Mean diff = {1}%'''.format(std_check, mean_check) 
            plt.figure()
            plt.hist(r_arr, bins = bins, alpha=0.5, density = True, label='real data')
            plt.hist(f_arr, bins = bins, alpha=0.5, density = True, label='synthetic data')
            plt.legend(loc='upper right')
            plt.title('Histogram of {}'.format(r))
            plt.xlabel(r, fontsize = 8)
            plt.ylabel('Norm frequency', fontsize = 8) 
            plt.xticks(fontsize = 8)
            plt.yticks(fontsize = 8)
            plt.figtext(0.33, -0.02, text, ha = 'right')
            plot_loc = os.path.join(path, 'Hist_of_{}.jpeg'.format(r))
            plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = 600)
            plt.show(block = False) 
    
    # comparison of data distribution using statistical methods 
    #========================================================================== 
    def data_check(self, dataframe1, dataframe2, path):
        
        """ 
        data_check(dataframe1, dataframe2, path)
        
        Check the similarity beteween the real and synthetic data using the 
        Kolmogorov-Smirnoff test to compare the cumulative distribution functions 
        of the dataseries. The P value list for all dataframe columns is generated 
        and called as self.pv_list.  
        
        Keyword arguments:  
            
        dataframe1 (pd.dataframe):  dataframe of real numbers (original dataframe)
        dataframe2 (pd.dataframe):  dataframe of fake numbers (synthetic dataframe)
        path (str):                 figures (.jpeg) folder path
        
        Returns:
            
        None
        
        """
        self.pv_list = []
        self.real_list = []
        self.fake_list = [] 
        for col1, col2 in zip(dataframe1.columns, dataframe2.columns):
            array = dataframe1[col1].values
            self.real_list.append(array)
            array = dataframe2[col2].values        
            self.fake_list.append(array)
        for (r, f, t) in tqdm(zip(self.real_list, self.fake_list, dataframe1.columns)):
            ry, rx = np.histogram(r, bins = 'auto')
            sy, sx = np.histogram(f, bins = 'auto')
            real_cumsum = np.cumsum(ry)
            fake_cumsum = np.cumsum(sy)
            norm_real_cumsum = [x/real_cumsum[-1] for x in real_cumsum]
            norm_fake_cumsum = [x/fake_cumsum[-1] for x in fake_cumsum]
            statistic, p_value = ks_2samp(real_cumsum, fake_cumsum, 
                                          alternative = 'two-sided')
            statistic = round(statistic, 2)
            p_value = round(p_value, 3)
            self.pv_list.append(p_value)
            text = '''Statistics = {0}%
                      P value = {1}'''.format(statistic, p_value)   
            plt.plot(rx[:-1], norm_real_cumsum, c = 'blue', label = 'real data')
            plt.plot(sx[:-1], norm_fake_cumsum, c = 'orange', label = 'synthetic data')
            plt.xlabel(t, fontsize = 8)
            plt.ylabel('Cumulative norm frequency', fontsize = 8) 
            plt.xticks(fontsize = 8)
            plt.yticks(fontsize = 8)
            plt.legend(loc='upper left')
            plt.title('CDF of {}'.format(t))
            plt.figtext(0.33, -0.02, text, ha = 'right')
            plot_loc = os.path.join(path, 'CDF_of_{}.jpeg'.format(t))
            plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = 600)
            plt.show(block = False) 
            
        self.desc_list = []        
        for f in self.pv_list:
            if f >= 0.05:
                desc = 'Generated distribution is not equal'
            else:
                desc = 'Generated distribution is equal'
            self.desc_list.append(desc)
            
