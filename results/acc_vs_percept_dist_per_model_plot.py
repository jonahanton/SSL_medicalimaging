import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fpath = './full_data.csv'
df_full_data = pd.read_csv(fpath, index_col = False)
dict_full_data = df_full_data.set_index('Models').to_dict()

def mkdir_p(new_path):
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(new_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(new_path):
            pass
        else: raise

def plot_dset_acc_vs_perceptual_dist(model, transfer_setting='few shot', architecture='average'):

    acc_data = {}
    perc_dist_data = {}

    for dset in ['shenzhencxr','montgomerycxr','bach','iChallengeAMD', 'iChallengePM','chexpert','stoic','diabetic retinopathy', 'chestx']:
 
        setting_dset = transfer_setting + ' ' + dset
        try:
            dset_acc_dict = dict_full_data[setting_dset]
            acc_data[dset] = dset_acc_dict[model]

            attentive_diffusion_dset = f'perceptual distance {architecture} {dset}'
            try:
                attentive_diffusion_dict = dict_full_data[attentive_diffusion_dset]
                perc_dist_data[dset] = attentive_diffusion_dict[model]
                
            except KeyError:
                print(f'No data for perceptual distance on {dset} dataset.')
        except KeyError:
            print(f'No data for {setting_dset}.')

    shared_models = acc_data.keys() and perc_dist_data.keys()
    dict_intersection = {k: (acc_data[k], perc_dist_data[k]) for k in shared_models}
    new_df = pd.DataFrame.from_dict(dict_intersection, orient='index')
    new_df = new_df.reset_index(level=0)
    new_df.columns = ['Datasets', 'acc', 'perceptual distance']

    sns.set_style("whitegrid")
    g = sns.lmplot(x='acc', y="perceptual distance", hue="Datasets", data=new_df, fit_reg=False, palette="tab10", scatter_kws={"s": 100})
    sns.regplot(x='acc', y="perceptual distance", data=new_df, scatter=False, ax=g.axes[0, 0])
    plt.ylabel(f"Perceptual Distance ({architecture})")
    plt.xlabel(f"{transfer_setting.title()} Accuracy")
    plt.title(model)
    #plt.show()
    save_results_to = f'acc_vs_percept_dist_per_model/{model}/'
    mkdir_p(save_results_to)
    plt.savefig(save_results_to + f'{transfer_setting} acc vs attentive diff {architecture}.jpg', bbox_inches = "tight")


models = ['simclrv1', 'mocov2', 'swav', 'byol','pirl',
        'supervised r50', 'supervised r18','supervised d121', 
        'mimic-chexpert lr=0.01','mimic-chexpert lr=0.1', 
        'mimic-chexpert lr=1.0','mimic-cxr r18', 'mimic-cxr d121']

for model in models:
    plot_dset_acc_vs_perceptual_dist(model, architecture = 'squeezenet')

#plot_dset_acc_vs_perceptual_dist('byol', 'few shot')