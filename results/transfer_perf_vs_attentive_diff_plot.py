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

def plot_dset_acc_vs_attentive_diffusion(transfer_setting, dset):
    '''
    dset_name in ['shenzhencxr','montgomerycxr','bach',
                  'iChallengeAMD', 'iChallengePM','chexpert',
                  'stoic','diabetic retinopathy', 
                  'chestx']
    '''
    
    setting_dset = transfer_setting + ' ' + dset
    try:
        dset_acc_dict = dict_full_data[setting_dset]
    
        attentive_diffusion_dset = f'attentive diffusion {dset}'
        try:
            attentive_diffusion_dict = dict_full_data[attentive_diffusion_dset]

            shared_models = dset_acc_dict.keys() and attentive_diffusion_dict.keys()
            dict_intersection = {k: (dset_acc_dict[k], attentive_diffusion_dict[k]) for k in shared_models}
            new_df = pd.DataFrame.from_dict(dict_intersection, orient='index')
            new_df = new_df.reset_index(level=0)
            new_df.columns = ['Models', 'acc', 'attentive diffusion']

            sns.set_style("whitegrid")
            markers=['P', 'P', 'P','P', 'P', 'H', 'H', 'H', 'P', 'P', 'P', 'P', 'P']  
            colors = ['cornflowerblue', 'royalblue', 'lightskyblue', 'deepskyblue', 'steelblue',
                    'orangered', 'lightcoral', 'firebrick',
                    'limegreen', 'forestgreen', 'darkgreen', 'springgreen', 'seagreen']
            g = sns.lmplot(x='acc', y="attentive diffusion", hue="Models", data=new_df, fit_reg=False, markers=markers, palette=colors, scatter_kws={"s": 150})
            sns.regplot(x='acc', y="attentive diffusion", data=new_df, scatter=False, ax=g.axes[0, 0])
            plt.ylabel(f"Attentive Diffusion")
            plt.xlabel(f"{transfer_setting.title()} Accuracy")
            plt.title(dset)
            save_results_to = f'acc_vs_attentive_diff/{dset}/'
            mkdir_p(save_results_to)
            plt.savefig(save_results_to + f'{transfer_setting} acc vs attentive diff_{dset}.jpg', bbox_inches = "tight")
        except KeyError:
            print(f'No data for attentive diffusion on {dset} dataset.')
    except KeyError:
        print(f'No data for {setting_dset}.')




for dset_name in ['shenzhencxr','montgomerycxr','bach','iChallengeAMD', 'iChallengePM','chexpert',
                  'stoic','diabetic retinopathy', 'chestx']:
    for transfer_setting in ['few shot', 'finetune', 'linear']:
        plot_dset_acc_vs_attentive_diffusion(transfer_setting, dset_name)