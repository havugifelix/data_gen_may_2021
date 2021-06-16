#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mne
import os
from os.path import abspath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency import psd_welch
from tqdm import tqdm
import itertools
from multiprocessing import Process
file_path="/home/kashraf/felix_hd/data_gen_may_2021/EEG data/Visual/"
filename=os.listdir(file_path)
path_montage="/home/kashraf/felix_hd/data_gen_may_2021/montage/"
montage=mne.channels.read_montage(path_montage+"//"+"neuroscan64ch.loc")
raw_data=[]
for file in tqdm(filename):
    files=mne.io.read_raw_cnt(file_path+"/"+file,montage=montage, preload=True,verbose=False);
    raw_data.append(files)

## Selecting channels to include
good_ch= mne.pick_channels(raw_data[0].info['ch_names'], include=[],
                        exclude=["EKG","EMG",'VEO','HEO','Trigger'])
mne.pick_info(raw_data[0].info,sel=good_ch,copy=False,verbose=False)
for f in tqdm(raw_data):
    mne.pick_info(f.info,sel=good_ch,copy=False)
    f.set_montage(montage)


# In[4]:


events= mne.events_from_annotations(raw_data[6])

event_id_recall={'12': 1, '14': 2, '16': 3, '18': 4}
event_id_encoding={"2":5,"4":6,"6":7,"8":8}
# event_id_answer={""}
event_id_answer={"51":9,"52":10}


# In[5]:


import multiprocessing
from tqdm import tqdm
data=dict()
for i in range(len(raw_data)):
    data[i+1]={
    2:mne.Epochs(raw_data[i], events[0], event_id=event_id_encoding)["2"].get_data(),
    4:mne.Epochs(raw_data[i], events[0], event_id=event_id_encoding)["4"].get_data(),
    6:mne.Epochs(raw_data[i], events[0], event_id=event_id_encoding)["6"].get_data(),
    8:mne.Epochs(raw_data[i], events[0], event_id=event_id_encoding)["8"].get_data(),
    }


# ### Generate bootstrap sampling ERPS
# - Specify number of trials for boot to calculate one erp
# - Total number or erps per subject per CL level ( number of iterations for erp calculation

# In[17]:


from tqdm import tqdm

n_trials= 20# Number of trials to be selected and be averaged out to give one ERP
n_erp= 1000 # Nmber of ERPS required per subject per CL level. So for cL2 we have 1000*11 Erps
# from multiprocessing import Manager
# manager=Manager()
# data_new=manager.dict()

def bootstrap(event,trials, n_trials,n_erps):
    erps=[]
    for i in range(n_erps):
        t=np.random.choice(trials,n_trials)
        erp = np.average(event[t],axis=0)
        erps.append(erp)
    return erps
for i in tqdm(range(len(data))):
    data[i+1]["erp2"]=bootstrap(data[i+1][2], data[i+1][2].shape[0],n_trials,n_erp)
    data[i+1]["erp4"]=bootstrap(data[i+1][4], data[i+1][4].shape[0],n_trials,n_erp)
    data[i+1]["erp6"]=bootstrap(data[i+1][6], data[i+1][6].shape[0],n_trials,n_erp)
    data[i+1]["erp8"]=bootstrap(data[i+1][8], data[i+1][8].shape[0],n_trials,n_erp)

## Transform our data into epoch array
# from tqdm import tqdm
# def bootstrap_process(start):
for sub in tqdm(data):
    for i in range(1000):
        data[sub]["erp2"][i]=np.reshape(data[sub]["erp2"][i],(1,64,176))
        data[sub]["erp2"][i]=mne.EpochsArray(data[sub]["erp2"][i],info=raw_data[0].info)

        data[sub]["erp4"][i]=np.reshape(data[sub]["erp4"][i],(1,64,176))
        data[sub]["erp4"][i]=mne.EpochsArray(data[sub]["erp4"][i],info=raw_data[0].info)

        data[sub]["erp6"][i]=np.reshape(data[sub]["erp6"][i],(1,64,176))
        data[sub]["erp6"][i]=mne.EpochsArray(data[sub]["erp6"][i],info=raw_data[0].info)

        data[sub]["erp8"][i]=np.reshape(data[sub]["erp8"][i],(1,64,176))
        data[sub]["erp8"][i]=mne.EpochsArray(data[sub]["erp8"][i],info=raw_data[0].info)


# ### Generate and Save PSd topomaps
# - Define a fx that takes a erp data and frequency band of interest and generate average PSD and their topomaps
# -  Save PSD to the memory

# In[38]:


def topo_generator(my_data,band,path):
    if band=="delta":
        cmap="Greens"
        fmin=1
        fmax=3
    elif band=="delta":
        cmap="Blues"
        fmin=4
        fmax=7
    elif band=="alpha":
        cmap="Reds"
        fmin=8
        fmax=12
    elif band=="beta":
        cmap="PuRd"
        fmin=13
        fmax=30
    else:
        fmin=31
        fmax=100
        cmap="RdBu"
    psd,freq= mne.time_frequency.psd_welch(my_data, fmin= fmin, fmax=fmax,n_fft=176)
    psd1= np.average(psd,axis=2).flatten()
    fig=mne.viz.plot_topomap( psd1,my_data.info,cmap= cmap, contours=False)
    fig[0].get_figure().savefig(path+".png")  


# In[ ]:


from matplotlib import cm
from multiprocessing import Manager
manager=Manager()
data_new=manager.dict()
data_new= data
path_data="/home/kashraf/felix_hd/data_gen_may_2021/topo_data_may_v1/gamma/cl8//"
# path_cl4_alpha="/home/spring2021/topomap_dataset/visual/recall/704ms/alpha/cl4//"
# path_cl8_beta=r"/home/spring2021/topomap_dataset/704ms/beta/cl8//"

def generation_process(start):
    for sub in tqdm(data_new):
        for j in tqdm(range(200)):
            my_data=data_new[sub]["erp8"][start+j]
            path=path_data+"sub_"+str(sub)+"_gamma_"+str(start+j)
            topo_generator(my_data,"gamma",path)
            


if __name__=="__main__":
    p1= Process(target=generation_process,args=(0,))
    p2= Process(target=generation_process,args=(200,))
    p3= Process(target=generation_process,args=(400,))
    p4= Process(target=generation_process,args=(600,))
    p5= Process(target=generation_process,args=(800,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
   
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    
    print("Done")
    
    
        


# In[16]:


len(range(1,10))

