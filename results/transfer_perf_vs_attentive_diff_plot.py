import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import isnan
import scipy

fpath = './full_data.csv'
df_full_data = pd.read_csv(fpath, index_col = False)
dict_full_data = df_full_data.set_index('Models').to_dict()

dset_names = {'shenzhencxr': 'Shenzhen-CXR',
            'montgomerycxr': 'Montgomery-CXR',
            'bach': 'BACH',
            'iChallengeAMD': 'iChallenge-AMD', 
            'iChallengePM': 'iChallenge-PM',
            'chexpert': 'CheXpert', 
            'stoic': 'STOIC',
            'diabetic retinopathy': 'Diabetic Retinopathy', 
            'chestx': 'ChestX'}

models = ["simclrv1","mocov2","swav","byol","pirl","supervised r50",
            "supervised r18","supervised d121","mimic-chexpert lr=0.01",
            "mimic-chexpert lr=0.1","mimic-chexpert lr=1.0","mimic-cxr r18",
            "mimic-cxr d121"] 

ssl_models = ["simclrv1","mocov2","swav","byol","pirl", "mimic-chexpert lr=1.0","mimic-chexpert lr=0.1","mimic-chexpert lr=0.01","mimic-cxr r18","mimic-cxr d121"] 
supervised_models = ["supervised r50", "supervised r18","supervised d121"]

# Plotting conventions we use across our work

markers = {model: 'P' for model in ssl_models}
markers.update({model: 'H' for model in supervised_models})
colors = {"simclrv1":'cornflowerblue', "mocov2":'royalblue', "swav":'lightskyblue', "byol":'deepskyblue', "pirl":'steelblue',
        "supervised r50":'orangered', "supervised r18":'lightcoral', "supervised d121":'firebrick',
        "mimic-chexpert lr=0.01":'limegreen', "mimic-chexpert lr=0.1":'forestgreen', "mimic-chexpert lr=1.0":'darkgreen', 
        "mimic-cxr r18":'springgreen', "mimic-cxr d121":'seagreen'}

def mkdir_p(new_path):
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(new_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(new_path):
            pass
        else: raise

def plot_dset_acc_vs_attentive_diffusion(transfer_setting, dset):
    
    setting_dset = transfer_setting + ' ' + dset
    if dset == 'diabetic retinopathy' or dset == 'chestx':
        setting_dset += ' (5way)'
        
    try:
        dset_acc_dict = dict_full_data[setting_dset]
    
        attentive_diffusion_dset = f'attentive diffusion {dset}'
        try:
            attentive_diffusion_dict = dict_full_data[attentive_diffusion_dset]

            shared_models = dset_acc_dict.keys() and attentive_diffusion_dict.keys()
            dict_intersection = {k: (dset_acc_dict[k], attentive_diffusion_dict[k]) for k in shared_models if not isnan(dset_acc_dict[k]) and not isnan(attentive_diffusion_dict[k])}
            new_df = pd.DataFrame.from_dict(dict_intersection, orient='index')
            new_df = new_df.reset_index(level=0)
            new_df.columns = ['Models', 'acc', 'attentive diffusion']

            sns.set_style("darkgrid")
            markers_style = [markers[model] for model in dict_intersection.keys()]
            g = sns.lmplot(x='acc', y="attentive diffusion", hue="Models", data=new_df, fit_reg=False, markers=markers_style, palette=colors, scatter_kws={"s": 100}, height=4)
            sns.regplot(x='acc', y="attentive diffusion", data=new_df, scatter=False, ax=g.axes[0, 0])
            r, p = scipy.stats.pearsonr(x=new_df['acc'], y=new_df['attentive diffusion'])
            ax = plt.gca()
            ax.text(.05, .8, 'r={:.2f} \np={:.2g}'.format(r, p), transform=ax.transAxes)
            plt.ylabel(f"Attentive Diffusion")
            plt.xlabel(f"{transfer_setting.title()} Accuracy")
            plt.title(dset_names[dset])
            save_results_to = f'acc_vs_attentive_diff/{dset}/'
            mkdir_p(save_results_to)
            plt.savefig(save_results_to + f'{transfer_setting} acc vs attentive diff_{dset}.jpg', bbox_inches = "tight")
        except KeyError:
            print(f'No data for attentive diffusion on {dset} dataset.')
    except KeyError:
        print(f'No transfer performance data for {setting_dset} on {dset}.')




for dset_name in ['shenzhencxr','montgomerycxr','bach','iChallengeAMD', 'iChallengePM','chexpert',
                  'stoic','diabetic retinopathy', 'chestx']:
    for transfer_setting in ['few shot', 'finetune', 'linear']:
        plot_dset_acc_vs_attentive_diffusion(transfer_setting, dset_name)