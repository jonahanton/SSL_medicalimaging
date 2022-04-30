import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fpath = './full_data.csv'
df_full_data = pd.read_csv(fpath, index_col = False)
dict_full_data = df_full_data.set_index('Models').to_dict()

# alexnet
fpath = './perceptual-distance/alexnet.csv'
df_alexnet = pd.read_csv(fpath)
alexnet_dict = df_alexnet.set_index('model').to_dict()

# squeezenet
fpath = './perceptual-distance/squeezenet.csv'
df_squeezenet = pd.read_csv(fpath)
squeezenet_dict = df_squeezenet.set_index('model').to_dict()

# vgg
fpath = './perceptual-distance/vgg.csv'
df_vgg = pd.read_csv(fpath)
vgg_dict = df_vgg.set_index('model').to_dict()


def mkdir_p(new_path):
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(new_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(new_path):
            pass
        else: raise

def retrieve_architecture(number):
    if number == 0:
        return 'AlexNet'
    elif number == 1:
        return 'VGG'
    elif number == 2:
        return 'SqueezeNet'
    return None

def plot_perceptual_distance(dataset):
    '''
    Args:
        dataset - str in ['shenzhen-cxr','montgomery-cxr','diabetic_retinopathy','chestx','bach','iChallenge-AMD','iChallenge-PM','chexpert', 'stoic', 'imagenet']

    '''
    dset_pd_alexnet = alexnet_dict[dataset]
    dset_pd_squeezenet = squeezenet_dict[dataset]
    dset_pd_vgg = vgg_dict[dataset]

    models = ['simclr-v1', 'moco-v2', 'swav', 'byol', 'pirl',
              'supervised_r50', 'supervised_r18', 'supervised_d121',
              'mimic-chexpert_lr_0.01', 'mimic-chexpert_lr_0.1', 'mimic-chexpert_lr_1.0',
              'moco-cxr_r18', 'moco-cxr_d121']

    #dict_pd = {}
    rows = []
    for i, dict in enumerate([dset_pd_alexnet, dset_pd_vgg, dset_pd_squeezenet]): 
        architecture = retrieve_architecture(i)
        for model in models:
            rows.append({'Model': model, 'architecture': architecture, 'perceptual_distance':dict[model]})
    new_df = pd.DataFrame.from_dict(rows, orient='columns')


    sns.set_style("whitegrid")
    markers=['P', 'P', 'P','P', 'P', 'H', 'H', 'H', 'P', 'P', 'P', 'P', 'P']  
    colors = ['cornflowerblue', 'royalblue', 'lightskyblue', 'deepskyblue', 'steelblue',
            'orangered', 'lightcoral', 'firebrick',
            'limegreen', 'forestgreen', 'darkgreen', 'springgreen', 'seagreen']
    g = sns.catplot(x='architecture', y="perceptual_distance", data=new_df, hue='Model', palette=colors, height = 3)
    plt.ylabel(f"Perceptual Distance")
    plt.title(f'{dataset.capitalize()} reconstruction')
    #plt.show()
    save_results_to = f'perceptual_distance_plots/{dataset}/'
    mkdir_p(save_results_to)
    plt.savefig(save_results_to + f'{dataset}_pd_plot.jpg', bbox_inches = "tight")

for dataset in ['shenzhen-cxr','montgomery-cxr','diabetic_retinopathy','chestx','bach','iChallenge-AMD','iChallenge-PM','chexpert']: #Need to change files to adapt to stoic and imagenet
    plot_perceptual_distance(dataset)