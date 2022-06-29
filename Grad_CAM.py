import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
# from numba import cuda

# image process
import os
import shutil
import PIL

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm



"""def release_GPUmemory():
    K.clear_session()
    cuda.select_device(0)
    cuda.close()
    """


# grad-CAM main algorithm
# code from https://keras.io/examples/vision/grad_cam/
# author: fchollet

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, save_dir, conv_layer_name, heatmap, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # only heatmap
    _jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap * alpha)
    #display(_jet_heatmap)

    # cam_image_name
    cam_path = img_path

    _cam_path = cam_path.split('/')
    _cam_path = _cam_path[-1]
    _cam_path = _cam_path.split('.')
    file_name = _cam_path[0]

    heatmap_img_save_dir = save_dir + file_name + '_' + conv_layer_name + '_' + 'heatmap' + ".png"
    CAM_img_save_dir = save_dir + file_name + '_' + conv_layer_name + '_' + 'GradCAM' + ".png"

    # Save the heatmap image
    _jet_heatmap.save(heatmap_img_save_dir, 'PNG')

    # Save the superimposed image
    superimposed_img.save(CAM_img_save_dir, 'PNG')

    # Display Grad CAM
    #display(superimposed_img)

    

def get_layer_name(model):

    # this will return all the conv_layers name in a list
    layers_name = []
    # <class: 'Conv2D'>
    type_Conv2D = type(keras.layers.Conv2D(1, kernel_size=(1,1)))

    for idx in range(len(model.layers)):
        if(isinstance(model.get_layer(index=idx), type_Conv2D)):
            layers_name.append(model.get_layer(index=idx).name)

    return layers_name


def mkdir(save_dir):
    
    if os.path.exists(save_dir):
        for i in os.listdir(save_dir):
            _sub_path = os.path.join(save_dir, i)

            if os.path.isfile(_sub_path):
                os.remove(_sub_path)
            if os.path.isdir(_sub_path):
                shutil.rmtree(_sub_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



model_builder = keras.applications.vgg16.VGG16(weights="imagenet")
img_size = (224, 224)
preprocess_input = keras.applications.vgg16.preprocess_input
decode_predictions = keras.applications.vgg16.decode_predictions

conv_layers_name = get_layer_name(model_builder)
#layer = model_builder.get_layer('block1_conv1')
#last_conv_layer_name = "block14_sepconv2_act"

# The local path to our target image
from os import walk

path = "ImageNet/generated_inputs/"
folders = next(walk(path))[1]

print(folders)


# # main


# generate a heatmap for original image


origin_img_dir = "ImageNet/seeds_20/"
origin_img_file = next(walk(origin_img_dir), (None, None, []))[2]

# make folder for save path
origin_save_dir = "ImageNet/Grad_CAM_output_originals/"
mkdir(origin_save_dir)

for file in origin_img_file:

    # fetch image
    origin_img_array = preprocess_input(get_img_array(origin_img_dir + file, size=img_size))

    # 
    #_tmp_data = np.squeeze(origin_img_array, axis=0).shape
    # save tmp_data
    #_tmp_image = PIL.Image.fromarray(_tmp_data)
    #_tmp_image.save('tmp.png')

    # fetch file name for folder
    _origin_img_name = file.split('.')  #  'ILSVRC2012_test_00000xxx', 'JPEG'
    _folder_name = _origin_img_name[0]  #  'ILSVRC2012_test_00000xxx'

    # make folder for image
    mkdir(origin_save_dir + _folder_name)

    # make model
    model = model_builder

    # model.predict(oringin_img_array)

    # generate and save heatmap (for original image)
    _save_dir = origin_save_dir + _folder_name + '/'

    for idx in range(len(conv_layers_name)):

        heatmap = make_gradcam_heatmap(origin_img_array, model, conv_layers_name[idx])

        # save heatmap
        save_and_display_gradcam(origin_img_dir + file, _save_dir, conv_layers_name[idx], heatmap)



# generate a heatmap for normal DLFuzz image
# initialize subdir name
# subdir = 0

for folder in folders:
    filenames = next(walk(path+folder+'/gen_img/'), (None, None, []))[2]  # [] if no file

    save_dir = 'ImageNet/Grad_CAM_output/normal/' + folder + '/'
    mkdir(save_dir)

    for file in filenames:
        # Prepare image
        img_array = preprocess_input(get_img_array(path+folder+'/gen_img/' + file, size=img_size))

        # get image id
        _adv_img_path = file.split('.')
        _adv_img_name = _adv_img_path[0]  # 00000xxx_adversarialPrediction_originalPrediction_timeStamp
        _adv_img_id = _adv_img_name.split('_')
        _adv_img_id = _adv_img_id[0] + '_' + _adv_img_id[-1]  # 00000xxx_timeStamp
        print(_adv_img_id)

        # Make model
        model = model_builder

        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Print what the top predicted class is
        preds = model.predict(img_array)
        print("Predicted:", decode_predictions(preds, top=1)[0])

        # save_dir = 'ImageNet/Grad_CAM_output/' + '0' + str(subdir) + '/'
        _save_dir = save_dir + _adv_img_name + '/'  # use only in the loop

        # mkdir
        mkdir(_save_dir)

        # Generate class activation heatmap
        for idx in range(len(conv_layers_name)):
            heatmap = make_gradcam_heatmap(img_array, model, conv_layers_name[idx])

            # Display heatmap
            #plt.matshow(heatmap)
            #plt.show()
            save_and_display_gradcam(path+folder+'/gen_img/' + file, _save_dir, conv_layers_name[idx], heatmap)

        # subdir += 1




# generate a heatmap for random noise image
# initialize subdir name
# subdir = 0
for folder in folders:
    filenames = next(walk(path+folder+'/random_img/'), (None, None, []))[2]  # [] if no file

    save_dir = 'ImageNet/Grad_CAM_output/random/' + folder + '/'
    mkdir(save_dir)

    for file in filenames:
        # Prepare image
        img_array = preprocess_input(get_img_array(path+folder+'/random_img/' + file, size=img_size))

        # get image id
        _adv_img_path = file.split('.')
        _adv_img_name = _adv_img_path[0]  # 00000xxx_adversarialPrediction_originalPrediction_timeStamp
        _adv_img_id = _adv_img_name.split('_')
        _adv_img_id = _adv_img_id[0] + '_' + _adv_img_id[-1]  # 00000xxx_timeStamp
        print(_adv_img_id)

        # Make model
        model = model_builder

        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Print what the top predicted class is
        preds = model.predict(img_array)
        print("Predicted:", decode_predictions(preds, top=1)[0])

        # save_dir = 'ImageNet/Grad_CAM_output/' + '0' + str(subdir) + '/'
        _save_dir = save_dir + _adv_img_name + '/'  # use only in the loop

        # mkdir
        mkdir(_save_dir)

        # Generate class activation heatmap
        for idx in range(len(conv_layers_name)):
            heatmap = make_gradcam_heatmap(img_array, model, conv_layers_name[idx])

            # Display heatmap
            #plt.matshow(heatmap)
            #plt.show()
            save_and_display_gradcam(path+folder+'/random_img/' + file, _save_dir, conv_layers_name[idx], heatmap)

        # subdir += 1

# release_GPUmemory()

