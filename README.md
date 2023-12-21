# TabularGAN

## Project description
TabularGAN is a tabular data generator based on a Generative Adversarial Network (GAN). This app works using .csv files as input to collect the data from, and generate synthetic tabular data with joint distribution starting from the empirical distribution of said data. Since the neural network model is based on the concept of GANs, it relies on the synergic activity of a generator and a discriminator model. Upon training of the models, the related model weights are saved for future use. While running, the script shows the accuracy and loss function output of both the generator and discriminator models, according to both the fake and real numbers dataset. The generated dataset is validated by plotting the superimposed histograms and cumulative distribution functions of the real and synthetic data. Moreover, the Kolgomorov-Smirnoff test is performed to validate the similarity between the real and fake distributions.

## Generative Adversarial Network (GAN)
A GAN consists of two main components: a generator and a discriminator. These two models work together in a sort of competition, where on oine hand the generator creates new data instances (trying to make them as realistic as possible), and on the other hand the discriminator examines the generated instances and tries to distinguish between real instances (from the input data) and fake instances (created by the generator). The generator and discriminator are thus trained together, with the generator trying to fool the discriminator and the discriminator trying to correctly classify instances as real or fake. This synergistic process leads to the generator producing increasingly realistic data over time. Once the models are trained, their weights are saved. This allows you to generate new data in the future without having to retrain the models, albeit considering the different data distribution for inference when compared to the training pool.

## How to use
Run the main python file TABGANGEN.py and use the GUI to navigate the various options. In the **Select folder paths** frame, you can select the source folder where your data are located and use the dropdown menu to select a specific file. At the bottom of the window, you can also select a folder where to save the processed data. In the main window you will find three main buttons:  

**Generative Adversarial Network (GAN) pretraining:** pretrain the GAN model to learn the jointed data distribution 

**Generate synthetic values using pretrained GAN:** generate synthetiuc values using the pretrained model

**Validate synthetic data:** validate goodness of the synthetic data pool with statisticl methods

### Requirements
This application has been developed and tested using the following dependencies (Python 3.10.12):

- `keras==2.10.0`
- `matplotlib==3.7.2`
- `numpy==1.25.2`
- `pandas==2.0.3`
- `PySimpleGUI==4.60.5`
- `scikit-learn==1.3.0`
- `scipy==1.11.2`
- `seaborn==0.12.2`
- `tensorflow==2.10.0`
- `tqdm==4.66.1`

## Graphic interface
Here is a snapshot of the main GUI window:

Coming soon

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
