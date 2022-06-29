import os 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import PIL
from IPython.display import Image, display
from scipy import spatial


def mkdir(save_dir):
    if os.path.exists(save_dir):
        for i in os.listdir(save_dir):
            path_file = os.path.join(save_dir, i)

            if os.path.isfile(path_file):
                os.remove(path_file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



dir_normal = "ImageNet/Grad_CAM_output/normal/"
dir_random = "ImageNet/Grad_CAM_output/random/"

save_dir_normal = 'ImageNet/Lowest_Highest_Sim/normal/'
save_dir_random = 'ImageNet/Lowest_Highest_Sim/random/'

root_directory = dir_normal
save_dir = save_dir_normal

mkdir(save_dir + 'high/')
mkdir(save_dir + 'low/')

folders = os.listdir(root_directory)
for folder in folders:
    print(folder)
    imgs = os.listdir(root_directory+folder+'/')
    for img in imgs:
        img = img.split('_')  
        img = img[0]




def get_images_from_layer(sim, cam1, cam2, heatmap1, heatmap2, perturb, folder, img, img_id, high_low):
    # create figure
    fig_cam = plt.figure(figsize=(25, 25))

    # setting values to rows and column variables
    rows_cam = 3
    columns_cam = 3
    counter = 1

    img1 = PIL.Image.open('ImageNet/Grad_CAM_output_originals/ILSVRC2012_test_' + img_id + '/' + cam1)
    img2 = PIL.Image.open(root_directory + folder + '/' + img + '/' + cam2)

    img1 = img1.resize((round(img2.size[0]), round(img2.size[1])))
    img2 = img2.resize((round(img2.size[0]), round(img2.size[1])))

    fig_cam.add_subplot(rows_cam, columns_cam, counter)
    counter = counter + 1
    plt.imshow(img1)
    plt.axis('off')
    plt.title(cam1)

    fig_cam.add_subplot(rows_cam, columns_cam, counter)
    counter = counter + 1
    plt.imshow(img2)
    plt.axis('off')
    plt.title(cam2)

    fig_cam.add_subplot(rows_cam, columns_cam, counter)
    counter = counter + 1
    plt.imshow(RGB_img3)
    plt.axis('off')
    plt.title("Perturbation")

    img1 = PIL.Image.open('ImageNet/Grad_CAM_output_originals/ILSVRC2012_test_' + img_id + '/' + heatmap1)
    img2 = PIL.Image.open(root_directory + folder + '/' + img + '/' + heatmap2)

    img1 = img1.resize((round(img2.size[0]), round(img2.size[1])))
    img2 = img2.resize((round(img2.size[0]), round(img2.size[1])))

    fig_cam.add_subplot(rows_cam, columns_cam, counter)
    counter = counter + 1
    plt.imshow(img1)
    plt.axis('off')
    plt.title(heatmap1)

    fig_cam.add_subplot(rows_cam, columns_cam, counter)
    counter = counter + 1
    plt.imshow(img2)
    plt.axis('off')
    plt.title(heatmap2)

    fig_cam.add_subplot(rows_cam, columns_cam, counter)
    counter = counter + 1
    plt.imshow(perturb)
    plt.axis('off')
    plt.title('Perturbation')
    
    layer_split = cam1.split('_')  #  'ILSVRC2012_test_00000xxx', 'JPEG'
    layer = layer_split[3] + layer_split[4]
    
    fig_cam.savefig(save_dir + high_low + '/' + layer + '_' + folder + '_' + img + '_' + str(sim) + '.png')


for folder in folders:
    imgs = os.listdir(root_directory+folder+'/')
    for img in imgs:
        # create figure
        img_split = img.split('_')  #  'ILSVRC2012_test_00000xxx', 'JPEG'
        img_id = img_split[0]
        fig = plt.figure(figsize=(25, 25))

        # setting values to rows and column variables
        rows = 1
        columns = 3

        gen_img = None
        perturb_img = None
        path_gen_img = "ImageNet/generated_inputs/"+folder+"/gen_img/"
        path_perturb = "ImageNet/generated_inputs/"+folder+"/perturb/"


        filenames = next(os.walk(path_gen_img), (None, None, []))[2] 
        for file in filenames:
            _img_path = file.split('_')
            _origin_img_id = _img_path[0]  # 0000000xxx

            if(_origin_img_id == img_id):
                gen_img = file


        filenames = next(os.walk(path_perturb), (None, None, []))[2] 
        for file in filenames:
            _img_path = file.split('_')
            _origin_img_id = _img_path[0]  # 0000000xxx
            if(_origin_img_id == img_id):
                perturb_img = file
                
    
        if gen_img != None and perturb_img != None:
            # reading images
            Image1 = cv2.imread('ImageNet/seeds_20/' + 'ILSVRC2012_test_' + img_id + '.JPEG')
            print(path_gen_img)
            print(gen_img)
            Image2 = cv2.imread(path_gen_img + gen_img)
            Image3 = cv2.imread(path_perturb + perturb_img)

            RGB_img1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
            RGB_img2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
            RGB_img3 = cv2.cvtColor(Image3, cv2.COLOR_BGR2RGB)

            # Adds a subplot at the 1st position
            fig.add_subplot(rows, columns, 1)

            # showing image
            plt.imshow(RGB_img1)
            plt.axis('off')
            plt.title(img_id + " (Original)")

            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)

            # showing image
            plt.imshow(RGB_img2)
            plt.axis('off')
            plt.title(img_id + " (DLFuzz Generated)")

            # Adds a subplot at the 3nd position
            fig.add_subplot(rows, columns, 3)

            # showing image
            plt.imshow(RGB_img3)
            plt.axis('off')
            plt.title(img_id + " (Perturbation)")

            cams = next(os.walk(root_directory+ folder + '/' + img), (None, None, []))[2] 
            cams_original = next(os.walk('ImageNet/Grad_CAM_output_originals/ILSVRC2012_test_' + img_id), (None, None, []))[2] 

            lowest_sim = 100
            highest_sim = 0

            for i in range(0,26):

                if "_heatmap.png" in cams[i]:
                    # calculate similarity
                    img1 = PIL.Image.open('ImageNet/Grad_CAM_output_originals/ILSVRC2012_test_' + img_id + '/' + cams_original[i])
                    img2 = PIL.Image.open(root_directory + folder + '/' + img + '/' + cams[i])

                    img1 = img1.resize((round(img2.size[0]), round(img2.size[1])))
                    img2 = img2.resize((round(img2.size[0]), round(img2.size[1])))

                    cam_array1 = np.array(img1)
                    cam_array2 = np.array(img2)

                    cam_array1 = cam_array1.flatten()
                    cam_array2 = cam_array2.flatten()

                    cam_array1 = cam_array1/255
                    cam_array2 = cam_array2/255

                    similarity = -1 * (spatial.distance.cosine(cam_array1, cam_array2) - 1)

                    if similarity < lowest_sim:
                        lowest_sim = similarity
                        lowest_heatmap1 = cams_original[i]
                        lowest_heatmap2 = cams[i]
                        lowest_cam1 = cams_original[i-1] 
                        lowest_cam2 = cams[i-1]

                    if similarity > highest_sim:
                        highest_sim = similarity
                        highest_heatmap1 = cams_original[i]
                        highest_heatmap2 = cams[i]
                        highest_cam1 = cams_original[i-1] 
                        highest_cam2 = cams[i-1]

            get_images_from_layer(lowest_sim, lowest_cam1, lowest_cam2, lowest_heatmap1, lowest_heatmap2, RGB_img3, folder, img, img_id, "low")
            get_images_from_layer(highest_sim, highest_cam1, highest_cam2, highest_heatmap1, highest_heatmap2, RGB_img3, folder, img, img_id, "high")

