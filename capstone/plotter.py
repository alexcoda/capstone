from scipy import interpolate
from collections import defaultdict as dd
import matplotlib.pyplot as plt
import csv
import numpy as np
import glob
from matplotlib.font_manager import FontProperties
import pandas as pd
import math


def get_arrays(df):
    

    if "phase" in set(list(df)):
        if df.loc[0].phase == -1.0:
            plot_type = "da"
        else:
            plot_type = "lwf"
    else:
        plot_type = "ewc"

    headers = [name for name in list(df) if name[:4]=='test']
    num_tasks = len(headers)
    results_map = dd(list)
    for i, row in df.iterrows():

        for header in headers:
            value = row[header]

            if ((np.isnan(value)) or (value<=0)):
                results_map[header].append(-1)
            else:
                results_map[header].append(value*100)
                
    l = sorted([value for key, value in results_map.items()], reverse=True)
    
    return plot_type, [arr for arr in l]

def get_dist(l, num_tasks, plot_type):
    
    if num_tasks == 2:
        
        cpt = 0 
        for i, element in enumerate(l[1]):
            if element<0:
                cpt += 1
            else:
                break
                
        if plot_type == "da":
            cpt = len(l[1])//2

        max_dist = max(cpt, len(l[1])-cpt)
        return [cpt], max_dist
    
    if num_tasks == 3:
        
        cpt1 = 0 
        for i, element in enumerate(l[1]):
            if element<0:
                cpt1 += 1
            else:
                break
                
        cpt2 = 0 
        for i, element in enumerate(l[2]):
            if element<0:
                cpt2 += 1
            else:
                break
                
        max_dist = max(max(cpt1, cpt2-cpt1), len(l[2])-cpt2)

        return [cpt1, cpt2], max_dist


def rearrange(arr, num_tasks, cpts, new_size, plot_type):

    if num_tasks == 2:
        
        cpt = cpts[0]
        arr1, arr2 = arr[:cpt], arr[cpt:]
        arr = resize(arr1, new_size) + resize(arr2, new_size)
        
        return arr
    
    if num_tasks == 3:
        
        cpt1, cpt2 = cpts
        arr1, arr2, arr3 = arr[:cpt1], arr[cpt1:cpt2], arr[cpt2:]
        arr = resize(arr1, new_size) + resize(arr2, new_size) + resize(arr3, new_size)
        
        return arr


def resize(y, size):
    
    if size == len(y):
        return y
    
    first, last = y[0], y[-1]
    y = np.array(y)
    x = np.arange(0, len(y))
    
    f = interpolate.interp1d(x, y)
    m = len(y)-1
    
    xnew = np.arange(0, m, m/size)
    ynew = f(xnew).tolist()
    
    return [first] + ynew[1:-1] + [last]  


def values_from_frames(frames):
    
    all_lists = []
    plot_types = []
    for df in frames:
        plot_type, l = get_arrays(df)
        all_lists.append(l)
        plot_types.append(plot_type)

    max_size = 0
    num_tasks = len(l)

    for plot_type, l in zip(plot_types, all_lists):
        cpts, max_dist = get_dist(l, num_tasks, plot_type)
        max_size = max(max_size, max_dist)

    new_all_lists = []
    for plot_type, l in zip(plot_types, all_lists):
        cpts, max_dist = get_dist(l, num_tasks, plot_type)

        arrs = []
        for arr in l:
            new_arr = rearrange(arr, num_tasks, cpts, max_size,  plot_type)
            arrs.append(new_arr)
        new_all_lists.append(arrs)

    values = [[] for i in range(num_tasks)]
    for l in new_all_lists:
        for i in range(len(l)):
            arr = l[i]
            arr = [(value if value>0 else -float(0)) for value in arr]
            arr = arr + [arr[-1]]
            values[i].append(arr)
    
    return values, max_size, num_tasks, plot_types


def ax_plot(axs, k, first, value_arr, num_tasks, colors, 
            task_names, train_names, labels, y_range,
            task_font, train_font, div, include_lwf, plot_types):

    for i, value in enumerate(value_arr):

        plot_type = plot_types[i]
        if plot_type == "lwf":
            first_half, second_half = value[:(len(value)//2)+1], value[(len(value)//2)+1:]
            half = (len(second_half)//2)
            divs = (half+1, half) if len(second_half)%2 else (half, half)
            fine, tune = second_half[:divs[0]], second_half[divs[1]:]
            fine, tune = resize(fine, len(second_half)), resize(tune, len(second_half))
            value = first_half + fine + tune
            first = [i for i in range(0, len(value))]

        axs[k].plot(first,value,color=colors[i], label=labels[i])
        axs[k].get_xaxis().set_ticks([])
        
        axs[k].set_ylabel(task_names[k], fontsize=task_font)
        axs[k].spines['bottom'].set_visible(False)
        axs[k].spines['top'].set_visible(False)
        box = axs[k].get_position()
        axs[k].set_position([box.x0, box.y0, box.width * 0.9, box.height])
 
    axs[k].set_ylim(y_range[0], y_range[1])

    if not include_lwf:
        axs[k].set_xlim(0, num_tasks*div)
        for i in range(1, num_tasks):
            axs[k].axvline(x = (i*div),linewidth=1, color='black', linestyle='--')
    else:
        axs[k].set_xlim(0, (num_tasks+1)*div)
        for i in range(1, num_tasks+1):
            axs[k].axvline(x = (i*div-1),linewidth=1, color='black', linestyle='--')
    
    #defining top labels (stages)
    if not k:
        
        axs[k].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if num_tasks == 2:
            axs[k].set_title(spacing(num_tasks, 0, train_names[0]), loc='left', fontsize=train_font)
            axs[k].set_title(spacing(num_tasks, 1, train_names[1]), loc='right', fontsize=train_font)
            
        if num_tasks == 3:
            axs[k].set_title(spacing(num_tasks, 0, train_names[0]), loc='left', fontsize=train_font)
            axs[k].set_title(spacing(num_tasks, 1, train_names[1]), loc='center', fontsize=train_font)
            axs[k].set_title(spacing(num_tasks, 2, train_names[2]), loc='right', fontsize=train_font)

        if include_lwf:
            axs[k].set_title(spacing(num_tasks, 0, train_names[0]), loc='left', fontsize=train_font)
            axs[k].set_title(spacing(num_tasks, 1, train_names[1]), loc='center', fontsize=train_font)
            axs[k].set_title(spacing(num_tasks, 2, "Fine Tune"), loc='right', fontsize=train_font)
            
            
def spacing(num_tasks, num, value):
    
    if num_tasks == 2:
        space = " "*5
        if num == 0:
            return space + value
        else:
            return value + space
        
    return value
            
def plot_all(num_tasks, colors, indexes, values, 
             task_names=['Task A', 'Task B', 'Task C'], train_names=['Train A', 'Train B', 'Train C'], 
             labels=['DA', 'LWF'], y_range=(40, 100), pdf_name="image",
             task_font=12, train_font=12, max_size=2, show=False, include_lwf=False, plot_types=[]):
    
    if not plot_types:
        raise Exception("No plot types added.")
    fig, axs = plt.subplots(num_tasks, 1, sharex=True)
    fig.subplots_adjust(hspace=0.2)
    title = 'Test'
    legend = ""

    for k in range(0, num_tasks):
        value_arr = values[k]
        ax_plot(axs, k, indexes, value_arr, num_tasks, colors, 
                task_names, train_names, labels, y_range,
                task_font, train_font, max_size, include_lwf, plot_types)
        
    plt.savefig(pdf_name+'.pdf')
    
    if show:
        plt.show()

def parse_options(options):

    print(options)
    
    if 'colors' not in options:
        options['colors'] = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:purple']
        
    if 'task_names' not in options:
        options['task_names'] = ['Task A', 'Task B', 'Task C']
        
    if 'train_names' not in options:
        options['train_names'] = ['Train A', 'Train B', 'Train C']
        
    if 'labels' not in options:
        options['labels'] = ['DA', 'LWF']
        
    if 'y_range' not in options:
        options['y_range'] = (40, 100)
        
    if 'pdf_name' not in options:
        options['pdf_name'] = 'example'
        
    if 'task_font' not in options:
        options['task_font'] = 12
                
    if 'train_font' not in options:
        options['train_font'] = 12
        
    if 'show' not in options:
        options['show'] = False
    
    return options

def plot_from_frames(dfs, options={}):
    
    values, max_size, num_tasks, plot_types = values_from_frames(dfs)
    indexes = [i for i in range(0, (num_tasks*max_size)+1)]
    options = parse_options(options)
    
    colors = options['colors']
    task_names = options['task_names']
    train_names = options['train_names']
    labels = options['labels']
    y_range = options['y_range']
    pdf_name = options['pdf_name']
    task_font = options['task_font']
    train_font = options['train_font']
    show = options['show']

    include_lwf = "lwf" in set(plot_types)

    plot_all(num_tasks, colors, indexes, values, 
             task_names, train_names, labels, y_range, pdf_name,
             task_font, train_font, max_size, show, include_lwf, plot_types)

def plot_from_csvs(names, options={}):
    
    dfs = [pd.read_csv(name) for name in names]
    plot_from_frames(dfs, options)