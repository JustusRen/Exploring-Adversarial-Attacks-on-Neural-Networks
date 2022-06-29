#!/usr/bin/env python
# coding: utf-8

# In[514]:


import os 
import shutil
import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee', 'no-latex'])
import numpy as np
import pandas as pd
import cv2
import re
import PIL
from IPython.display import Image, display
from scipy import spatial
from keras import backend as K
# from numba import cuda


# In[515]:


"""def release_GPUmemory():
    K.clear_session()
    cuda.select_device(0)
    cuda.close()"""


# In[516]:


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


# In[517]:


gen_dir = "ImageNet/generated_inputs/"
cam_dir_normal = "ImageNet/Grad_CAM_output/normal/"
cam_dir_random = "ImageNet/Grad_CAM_output/random/"

save_dir = 'ImageNet/Experiment_cos_sim/'
mkdir(save_dir)


# In[518]:


# input_dir = "ImageNet/Grad_CAM_output/xxxxx/"
def get_img_sim(cam_input_dir, fuzz_status):
    cos_sim_dict = {}

    _cam_perturb_lv_list = os.listdir(cam_input_dir)
    for _cam_perturb_lv in _cam_perturb_lv_list:
        _cam_dir_list = os.listdir(cam_input_dir + _cam_perturb_lv)
        if(_cam_dir_list != []):
            for _imgs_dir in _cam_dir_list:

                gen_img = None
                perturb_img = None
                
                _imgs_dir_split = _imgs_dir.split('_')
                _img_id = _imgs_dir_split[0]
                
                if(fuzz_status == 'normal'):
                    # get gen_filenames (adversarial samples)
                    normal_gen_filenames = next(os.walk(gen_dir + _cam_perturb_lv + '/gen_img/'), (None, None, []))[2]
                    # get gen_img (adversarial samples)
                    for file in normal_gen_filenames:
                        _img_path = file.split('_')
                        _origin_img_id = _img_path[0]  # 0000000xxx

                        if(_origin_img_id == _img_id):
                            gen_img = '/gen_img/' + file  # gen_img = "/gen_img/00000xxx_label1_label2_stamp.png"
                            break

                    # perturbation filenames
                    normal_perturb_filenames = next(os.walk(gen_dir + _cam_perturb_lv + '/perturb/'), (None, None, []))[2]
                    # perturbation image 
                    for file in normal_perturb_filenames:
                        _img_path = file.split('_')
                        _origin_img_id = _img_path[0]  # 0000000xxx

                        if(_origin_img_id == _img_id):
                            perturb_img = '/perturb/' + file
                            break

                    
                elif(fuzz_status == 'random'):
                    # get gen_filenames (adversarial samples)
                    random_gen_filenames = next(os.walk(gen_dir + _cam_perturb_lv + '/random_img/'), (None, None, []))[2]
                    # get gen_img (adversarial samples)
                    for file in random_gen_filenames:
                        _img_path = file.split('_')
                        _origin_img_id = _img_path[0]  # 0000000xxx

                        if(_origin_img_id == _img_id):
                            gen_img = '/random_img/' + file
                            break

                    # perturbation filenames
                    random_perturb_filenames = next(os.walk(gen_dir + _cam_perturb_lv + '/random_perturb/'), (None, None, []))[2]
                    # perturbation image 
                    for file in random_perturb_filenames:
                        _img_path = file.split('_')
                        _origin_img_id = _img_path[0]  # 0000000xxx

                        if(_origin_img_id == _img_id):
                            perturb_img = '/random_perturb/' + file
                            break

                # reading images
                Image1 = cv2.imread('ImageNet/seeds_20/' + 'ILSVRC2012_test_' + _img_id + '.JPEG')
                Image2 = cv2.imread(gen_dir + _cam_perturb_lv + gen_img)
                Image3 = cv2.imread(gen_dir + _cam_perturb_lv + perturb_img)

                RGB_img1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
                RGB_img2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
                RGB_img3 = cv2.cvtColor(Image3, cv2.COLOR_BGR2RGB)

                # Adds a subplot at the 1st position
                #fig.add_subplot(rows, columns, 1)
                # showing image
                #plt.imshow(RGB_img1)
                #plt.axis('off')
                #plt.title(_img_id + " (Original)")

                # Adds a subplot at the 2nd position
                #fig.add_subplot(rows, columns, 2)
                # showing image
                #plt.imshow(RGB_img2)
                #plt.axis('off')
                #plt.title(_img_id + " (DLFuzz Generated)")

                # Adds a subplot at the 3rd position
                #fig.add_subplot(rows, columns, 3)
                # showing image
                #plt.imshow(RGB_img1)
                #plt.axis('off')
                #plt.title(_img_id + " (Perturbation)")

                cams = next(os.walk(cam_input_dir + _cam_perturb_lv + '/' + _imgs_dir), (None, None, []))[2]
                cams_original = next(os.walk('ImageNet/Grad_CAM_output_originals/ILSVRC2012_test_' + _img_id), (None, None, []))[2]

                cos_sim_list = []

                for idx in range(0, 26):
                    # only work on heatmap
                    if(idx % 2 == 1):
                        # calculate similarity
                        img1 = PIL.Image.open('ImageNet/Grad_CAM_output_originals/ILSVRC2012_test_' + _img_id + '/' + cams_original[idx])
                        img2 = PIL.Image.open(cam_input_dir + _cam_perturb_lv + '/' + _imgs_dir + '/' + cams[idx])

                        # get layer name
                        _img_name = cams_original[idx].split('_')
                        _layer_name = _img_name[3] + " " + _img_name[4]

                        img1 = img1.resize((round(img2.size[0]), round(img2.size[1])))
                        img2 = img2.resize((round(img2.size[0]), round(img2.size[1])))

                        cam_array1 = np.array(img1)
                        cam_array2 = np.array(img2)

                        cam_array1 = cam_array1.flatten()
                        cam_array2 = cam_array2.flatten()

                        cam_array1 = cam_array1/255
                        cam_array2 = cam_array2/255

                        similarity = -1 * (spatial.distance.cosine(cam_array1, cam_array2) - 1)
                        cos_sim_list.append(similarity)

                        # create figure (subfigure)



                # dict["path"] = plt_sim_list
                cos_sim_dict[cam_input_dir + _cam_perturb_lv + '/' + _imgs_dir] = cos_sim_list

    return cos_sim_dict
    


# In[519]:


def get_random_img(cam_input_dir):
    ...


# In[520]:


# input_folder = "ImageNet/Grad_CAM_output/xxxxx/"
def calculate_cos_sim(cam_input_dir):

    _input_dir_split = cam_input_dir.split('/')
    _fuzz_status = _input_dir_split[-2]
    # get sim value, return dict
    cos_sim_dict = get_img_sim(cam_input_dir, _fuzz_status)

    # print(cos_sim_dict)
    # return a dict
    return cos_sim_dict


# In[521]:


def data_process_layer(_input):
    layer_data = {"Block1_Conv1": [], "Block1_Conv2": [], "Block2_Conv1": [], "Block2_Conv2": [], "Block3_Conv1": [], "Block3_Conv2": [], "Block3_Conv3": [], "Block4_Conv1": [], "Block4_Conv2": [], "Block4_Conv3": [], "Block5_Conv1": [], "Block5_Conv2":[], "Block5_Conv3":[]}
    
    # counter
    idx_count = 0
    for layer in layer_data:
        for element in _input:
            if(idx_count >= 13):
                break
            else:
                # _layer_data_list is a list of all cos values of the image
                _layer_data_list = _input[element]
                layer_data[layer].append(_layer_data_list[idx_count])

        idx_count += 1

    # print(layer_data)
    
    return layer_data
    


# In[522]:


def data_process_perturb(_input):
    perturb_data = {"perturb_level_0_25":{}, "perturb_level_0_5": {}, "perturb_level_1":{}, "perturb_level_2":{}, "perturb_level_4":{}}
    
    # couter
    for element in _input:
        _element_split = element.split('/')
        _perturb_lv = _element_split[3]

        perturb_data[_perturb_lv][element] = _input[element]
    
    return perturb_data


# In[523]:


def draw_boxplot(input, fuzz_status):
    dataframe = pd.DataFrame(input)
    #print(dataframe)

    columns = get_dataframe_column(dataframe)

    fig, ax = plt.subplots()
    ax.boxplot(columns)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    ['1-1', '1-2', '2-1', '2-2', '3-1', '3-2', '3-3', '4-1', '4-2', '4-3', '5-1', '5-2', '5-3'], 
    rotation=45)
    
    #if(fuzz_status == "normal"):
    #    ax.set_title("cosine similarity on adversial samples\' layers")
    #elif(fuzz_status == "random"):
    #    ax.set_title("cosine similarity on\n random noise perturbation samples\' layers")

    ax.set_ybound(0.5, 1)

    ax.set_xlabel("layers")
    ax.set_ylabel("cos similarity")
    plt.show()

    fig.savefig(save_dir + "boxplot_" + fuzz_status + '.pdf')
    


# In[524]:


def draw_boxplot_bylv():
    ...


# In[525]:


def draw_plot(_input, color, style, perturb_lv, ax):
    if(_input.empty):
        return
    else:
        columns = get_dataframe_column(_input)
        ax.plot(columns, c = color, linestyle = style, label = "perturb level = " + perturb_lv)
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
        ['1-1', '1-2', '2-1', '2-2', '3-1', '3-2', '3-3', '4-1', '4-2', '4-3', '5-1', '5-2', '5-3'], 
        rotation=45)
        ax.set_ybound(0.5, 1)
        
    


# In[526]:


def draw_line_plot(_input, fuzz_status):
    fig, ax = plt.subplots()
    ax.set_xlabel("layers")
    ax.set_ylabel("cos similarity")

    # input: nested dict
    perturb_lv_0_25 = data_process_layer(_input['perturb_level_0_25'])
    perturb_lv_0_5 = data_process_layer(_input["perturb_level_0_5"])
    perturb_lv_1 = data_process_layer(_input["perturb_level_1"])
    perturb_lv_2 = data_process_layer(_input["perturb_level_2"])
    perturb_lv_4 = data_process_layer(_input["perturb_level_4"])

    dataframe_0_25 = get_avg(perturb_lv_0_25)
    dataframe_0_5 = get_avg(perturb_lv_0_5)
    dataframe_1 = get_avg(perturb_lv_1)
    dataframe_2 = get_avg(perturb_lv_2)
    dataframe_4 = get_avg(perturb_lv_4)

    draw_plot(dataframe_0_25, 'red', '-', "0.25", ax)
    draw_plot(dataframe_0_5, 'blue', '--', "0.5", ax)
    draw_plot(dataframe_1, 'black', '-.', "1", ax)
    draw_plot(dataframe_2, 'green', ':', "2", ax)
    draw_plot(dataframe_4, 'yellow', '-.', "4", ax)

    ax.legend(loc = 'lower right')

    plt.show()

    fig.savefig(save_dir + "lineplot_" + fuzz_status + '.pdf')


# In[527]:


def get_avg(_input):
    # _input is a dict
    # return a dataframe
    for element in _input:
        if(_input[element] != []):
            data_array = _input[element]
            avg = []
            avg.append(sum(data_array) / len(data_array))
            _input.update({element: avg})
    #print(_input)

    dataframe = pd.DataFrame(_input)
    return dataframe
    


# In[528]:


def get_dataframe_column(dataframe):

    if(dataframe.empty):
        return None
    else:
        Block1_Conv1 = dataframe['Block1_Conv1']
        Block1_Conv2 = dataframe['Block1_Conv2']
        Block2_Conv1 = dataframe['Block2_Conv1']
        Block2_Conv2 = dataframe['Block2_Conv2']
        Block3_Conv1 = dataframe['Block3_Conv1']
        Block3_Conv2 = dataframe['Block3_Conv2']
        Block3_Conv3 = dataframe['Block3_Conv3']
        Block4_Conv1 = dataframe['Block4_Conv1']
        Block4_Conv2 = dataframe['Block4_Conv2']
        Block4_Conv3 = dataframe['Block4_Conv3']
        Block5_Conv1 = dataframe['Block5_Conv1']
        Block5_Conv2 = dataframe['Block5_Conv2']
        Block5_Conv3 = dataframe['Block5_Conv3']

        columns = [Block1_Conv1, Block1_Conv2, Block2_Conv1, Block2_Conv2, Block3_Conv1, Block3_Conv2, Block3_Conv3, Block4_Conv1, Block4_Conv2, Block4_Conv3, Block5_Conv1, Block5_Conv2, Block5_Conv3]
        return columns


# In[529]:


def calc_vulnerablility(perturb_data_normal, perturb_data_random):
    random_perturb_lv_0_25 = data_process_layer(perturb_data_random['perturb_level_0_25'])
    random_perturb_lv_0_5 = data_process_layer(perturb_data_random["perturb_level_0_5"])
    random_perturb_lv_1 = data_process_layer(perturb_data_random["perturb_level_1"])
    random_perturb_lv_2 = data_process_layer(perturb_data_random["perturb_level_2"])
    random_perturb_lv_4 = data_process_layer(perturb_data_random["perturb_level_4"])
    
    random_dataframes = []
    random_dataframes.append(get_avg(random_perturb_lv_0_25))
    random_dataframes.append(get_avg(random_perturb_lv_0_5))
    random_dataframes.append(get_avg(random_perturb_lv_1))
    random_dataframes.append(get_avg(random_perturb_lv_2))
    random_dataframes.append(get_avg(random_perturb_lv_4))
    
    
    normal_perturb_lv_0_25 = data_process_layer(perturb_data_normal['perturb_level_0_25'])
    normal_perturb_lv_0_5 = data_process_layer(perturb_data_random["perturb_level_0_5"])
    normal_perturb_lv_1 = data_process_layer(perturb_data_random["perturb_level_1"])
    normal_perturb_lv_2 = data_process_layer(perturb_data_random["perturb_level_2"])
    normal_perturb_lv_4 = data_process_layer(perturb_data_random["perturb_level_4"])
    
    normal_dataframes = []
    normal_dataframes.append(get_avg(normal_perturb_lv_0_25))
    normal_dataframes.append(get_avg(normal_perturb_lv_0_5))
    normal_dataframes.append(get_avg(normal_perturb_lv_1))
    normal_dataframes.append(get_avg(normal_perturb_lv_2))
    normal_dataframes.append(get_avg(normal_perturb_lv_4))   
    
    vul_probas = []
    for i in range(0,5):
        
        random_avg = random_dataframes[i].stack().mean()
        normal_stack = normal_dataframes[i].stack()
        vul_counter = 0
        general_counter = 0
        for sim in normal_stack:
            general_counter += 1
            if sim < random_avg:
                vul_counter += 1
        """if normal_dataframes[i]["Block1_Conv1"] < random_avg[i]:
            counter += 1
        print(normal_dataframes[i]["Block1_Conv1"])"""
        vul_probas.append(vul_counter / general_counter)
    return vul_probas


# In[530]:


cos_sim_normal = calculate_cos_sim(cam_dir_normal)
layer_data_normal = data_process_layer(cos_sim_normal)
draw_boxplot(layer_data_normal, "normal")
perturb_data_normal = data_process_perturb(cos_sim_normal)
draw_line_plot(perturb_data_normal, "normal")


cos_sim_random = calculate_cos_sim(cam_dir_random)
layer_data_random = data_process_layer(cos_sim_random)
draw_boxplot(layer_data_random, "random")
perturb_data_random = data_process_perturb(cos_sim_random)
draw_line_plot(perturb_data_random, "random")

# release_GPUmemory()


# In[531]:


vul_probas = calc_vulnerablility(perturb_data_normal, perturb_data_random)


# In[532]:


print("Vul. Proba. for Attack Level 0.25: " + str(vul_probas[0]))
print("Vul. Proba. for Attack Level 0.5: " + str(vul_probas[1]))
print("Vul. Proba. for Attack Level 1: " + str(vul_probas[2]))
print("Vul. Proba. for Attack Level 2: " + str(vul_probas[3]))
print("Vul. Proba. for Attack Level 4: " + str(vul_probas[4]))

