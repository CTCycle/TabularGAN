# TabularGAN

## Project description
TabularGAN is a tabular data generator based on the Generative Adversarial Network (GAN) approach. This app allows extracting the data from .cvs files and geenerate synthetic tabular data with joint distribution starting from the empirical input data and its distribution. Sinc ethe neural network model is based on the concept of GANs, it exploits the the synergic activity of a generator and a discriminator model. Upon training of the models, the related model weights are saved for future use. While running, the script shows the accuracy and loss function output of both the generator and discriminator models, according to both the fake and real numbers dataset. The generated dataset is validated by plotting the superimposed histograms and cumulative distribution functions of the real and synthetic data. Moreover, the Kolgomorov-Smirnoff test is performed to validate the similarity between the real and fake distributions.

## How to use
Run the main python file TABGANGEN.py and use the GUI to navigate the various options.  

