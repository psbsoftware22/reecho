from preprocessing import img_preprocessing, show_image
from variational_autoencoder import VAE
from autoencoder_analysis import select_images, plot_reconstructed_images, plot_add_image

from sklearn.model_selection import train_test_split
import numpy as np


class TestModel:
    def __init__(self, path, normal_model, noisefilter_model, colorimage_model):
        self.normal_model = normal_model
        self.noisefilter_model = noisefilter_model
        self.colorimage_model = colorimage_model
        
        self.x_train, self.x_test= None, None
        self.x_train_grayscale, self.x_test_grayscale = None, None
        self.RAW_IMAGES = path
        self.X = None
        self.X_G = None
        
        
        
    def prerun(self):
        self.image_processing()
        self.prepare_datasets()
        
    def run(self):
        self.prerun()
        self.ImageReconstruction()
        self.NoiseReduction()
        self.NewImageGeneration()
        self.ColoringImage()


    def image_processing(self):
        self.X = img_preprocessing(self.RAW_IMAGES, zoom=60, resolution=[32, 32])
        show_image(self.X[6])
        print(self.X.shape)

        self.X_G = img_preprocessing(self.RAW_IMAGES, zoom=60, resolution=[32, 32], grayScale=True)
        show_image(self.X_G[6])
        print(self.X.shape)

    def prepare_datasets(self):
        self.x_train, self.x_test= train_test_split(self.X, test_size=0.1, random_state=42)
        self.x_train_grayscale, self.x_test_grayscale = train_test_split(self.X_G, test_size=0.1, random_state=42)



    def ImageReconstruction(self):
        normal_autoencoder = VAE.load(self.normal_model)
        num_sample_images_to_show = 5
        sample_images, _ = select_images(self.x_test, self.x_test, num_sample_images_to_show)
        reconstructed_images, _ = normal_autoencoder.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images, './test/exported-data/original-reconstructed', 'original-reconstructed.png', True)
        
    def NoiseReduction(self):
        noisefilter_autoencoder = VAE.load(self.noisefilter_model)
        x_test_noise =  self.x_test + np.random.normal(loc=0.0, scale=0.1, size=self.x_test.shape)
        num_sample_images_to_show = 5
        sample_images, _ = select_images(x_test_noise, x_test_noise, num_sample_images_to_show)
        reconstructed_images, _ = noisefilter_autoencoder.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images, './test/exported-data/noisefilter', 'original-reconstructed.png', True)
        
    
    def NewImageGeneration(self):
        addingimage_autoencoder = VAE.load(self.normal_model)
        image1 = []
        image1.append(self.X[0].reshape(1, self.X.shape[2], self.X.shape[2], 3))
        image2 = []
        image2.append(self.X[6].reshape(1, self.X.shape[2], self.X.shape[2], 3))

        reconstructed_image1, latent_space1 = addingimage_autoencoder.reconstruct(image1)
        reconstructed_image2, latent_space2 = addingimage_autoencoder.reconstruct(image2)

        reconstructed_images = addingimage_autoencoder.reconstruct_from_latentspace(latent_space1 + latent_space2)
        plot_add_image(image1[0], image2[0], reconstructed_images[0], './test/exported-data/generate_image', 'generate_image.png', True)
        
    def ColoringImage(self):
        colorimage_autoencoder = VAE.load(self.colorimage_model)
        num_sample_images_to_show = 5
        sample_images, _ = select_images(self.x_test_grayscale, self.x_test_grayscale, num_sample_images_to_show)
        reconstructed_images, _ = colorimage_autoencoder.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images, './test/exported-data/colorimage', 'colorimage.png', True,)
        
        
if __name__ == "__main__":
    normal_model = 'normal-model'
    noisefilter_model = 'noisefilter-model'
    colorimage_model = 'colorimage-model'
    testmodel = TestModel("./data/lfw", normal_model, noisefilter_model, colorimage_model)
    testmodel.run()