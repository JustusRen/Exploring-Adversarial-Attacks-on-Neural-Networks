#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import matplotlib.pyplot as plt


# In[35]:


def count_layers(layers):
    counts = {layer:layers.count(layer) for layer in layers}

    print('block1conv1: '+ str(counts.get('block1conv1')))
    print('block1conv2: '+ str(counts.get('block1conv2')))
    print('block2conv1: '+ str(counts.get('block2conv1')))
    print('block2conv2: '+ str(counts.get('block2conv2')))
    print('block3conv1: '+ str(counts.get('block3conv1')))
    print('block3conv2: '+ str(counts.get('block3conv2')))
    print('block3conv3: '+ str(counts.get('block3conv3')))
    print('block4conv1: '+ str(counts.get('block4conv1')))
    print('block4conv2: '+ str(counts.get('block4conv2')))
    print('block4conv3: '+ str(counts.get('block4conv3')))
    print('block5conv1: '+ str(counts.get('block5conv1')))
    print('block5conv2: '+ str(counts.get('block5conv2')))
    print('block5conv3: '+ str(counts.get('block5conv3')))
    return counts


# ### Get most vulnerable layer (normal)

# In[36]:


files = os.listdir('ImageNet/Lowest_Highest_Sim/normal/low/')
layers = []
for file in files:
    file_split = file.split('_')
    layer = file_split[0]
    layers.append(layer)
counts = count_layers(layers)
plt.title("Layer with lowest similarity")
plt.bar(counts.keys(), counts.values())


# ### Get most stable layer (normal)

# In[37]:


files = os.listdir('ImageNet/Lowest_Highest_Sim/normal/high/')
layers = []
for file in files:
    file_split = file.split('_')
    layer = file_split[0]
    layers.append(layer)
counts = count_layers(layers)
plt.title("Layer with highest similarity")
plt.bar(counts.keys(), counts.values())


# ### Get most vulnerable layer (random)

# In[38]:


files = os.listdir('ImageNet/Lowest_Highest_Sim/random/low/')
layers = []
for file in files:
    file_split = file.split('_')
    layer = file_split[0]
    layers.append(layer)
counts = count_layers(layers)
plt.title("Layer with lowest similarity (random noise)")
plt.bar(counts.keys(), counts.values())


# ### Get most stable layer (random)

# In[39]:


files = os.listdir('ImageNet/Lowest_Highest_Sim/random/high/')
layers = []
for file in files:
    file_split = file.split('_')
    layer = file_split[0]
    layers.append(layer)
counts = count_layers(layers)
plt.title("Layer with highest similarity (random noise)")
plt.bar(counts.keys(), counts.values())

