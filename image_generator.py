from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')



def generator(load_path, number=10):

    number = number - 1
    for subdir, dirs, files in os.walk(load_path):
        for file in files:
            img = load_img(os.path.join(subdir, file))  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            # Crop only faces and resize it
                # and saves the results to the `preview/` directory
            i = 0
            save_path = subdir
            #save_path = subdir.replace('lfw', 'lfw-generated')
            create_folder_if_it_doesnt_exist(save_path)
            for batch in datagen.flow(x, batch_size=1,
                            save_to_dir=save_path, save_prefix='new', save_format='jpg'):
                i += 1
                if i > number:
                    break  # otherwise the generator would loop indefinitely
    # the .flow() command below generates batches of randomly transformed images
    

def create_folder_if_it_doesnt_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

if __name__ == "__main__":
    RAW_IMAGES_NAME = "reecho/data/lfw-generated"
    X = generator(RAW_IMAGES_NAME, 2)

