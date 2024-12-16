"""
Visualizing point clouds
"""
from helper_tool import Plot
from helper_ply import read_ply
import numpy as np
import os

basedir = " "
vis_list=os.listdir(basedir)
plot_colors =[
    (0,0,0),  # green
     (190,190,190)]  # red
Flag=True
for name in vis_list:
     pred_path=basedir+"/"+name
     original_path=" "+name
     print(pred_path)
     pred_data = read_ply(pred_path)
     pred = np.vstack((pred_data['pred'])).astype(np.int32)
     label = np.vstack((pred_data['label']))
     original_data = read_ply(original_path)
     xyz = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T
     Plot.draw_pc_sem_ins(xyz,pred,name,plot_colors)
