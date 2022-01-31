# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:09:12 2022

@author: HP
"""

from fastai.vision.all import *
import os 
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
#
if __name__ ==  '__main__':
    p = os.path.join(os.getcwd(),'img_model_1.pkl')
    learn.fine_tune(1)
    learn.export(p)
