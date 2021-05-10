import numpy as np
import matplotlib.pyplot as plt
import os

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images, folder, name, saveplot=False, figsize=[25, 10]):
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.set_title('Original')
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.set_title('Reconstruncted')
        ax.imshow(reconstructed_image, cmap="gray_r")
    if (saveplot):  
        save_plot(plt, folder, name)
    plt.show()

def plot_add_image(image1, image2, reconstructed_image, folder, name, saveplot=False, figsize=[20, 10]):
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    image = image1.squeeze()
    ax = fig.add_subplot(2, 2, 1)
    ax.axis("off")
    ax.set_title('Original 1')
    ax.imshow(image, cmap="gray_r")
    image = image2.squeeze()
    ax = fig.add_subplot(2, 1, 1)
    ax.axis("off")
    ax.set_title('Original 2')
    ax.imshow(image, cmap="gray_r")
    reconstructed_image = reconstructed_image.squeeze()
    ax = fig.add_subplot(2, 2, 2)
    ax.axis("off")
    ax.set_title('Reconstruncted')
    ax.imshow(reconstructed_image, cmap="gray_r")
    if (saveplot):  
        save_plot(plt, folder, name)
    plt.show()
    return plt

def save_plot(plt, folder, name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, name))


