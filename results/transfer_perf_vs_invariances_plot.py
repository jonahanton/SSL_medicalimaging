import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Linear plots of the cosine similarity plotted against the transfer performance (accuracy)

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

def plot_dset_acc_vs_invariance(transfer_setting, dset):
    '''
    transfer_setting in ['few shot', 'finetune', 'linear']
    dset_name in ['shenzhencxr','montgomerycxr','bach',
                  'iChallengeAMD', 'iChallengePM','chexpert',
                  'stoic','diabetic retinopathy (5way)', 
                  'chestx (5way)', 'cifar10 (2way)']
    '''
    
    setting_dset = transfer_setting + ' ' + dset
    try:
        dset_acc_dict = dict_full_data[setting_dset]

        for invariance_type in ["rotation", "hflip", "hue", "translation", "multiview"]:
    
            invariance_type_dset = f'invariance {invariance_type} {dset}'
            try:
                invariance_dict = dict_full_data[invariance_type_dset]

                shared_models = dset_acc_dict.keys() and invariance_dict.keys()
                dict_intersection = {k: (dset_acc_dict[k], invariance_dict[k]) for k in shared_models}
                new_df = pd.DataFrame.from_dict(dict_intersection, orient='index')
                new_df = new_df.reset_index(level=0)
                new_df.columns = ['Models', 'acc', 'invariance']

                sns.set_style("whitegrid")
                # markers=['o', 'h', 'p','s', 'H', 'X', '*', 'P', '>', 'v', '^', '<', 'd'] 
                # models = ["simclrv1","mocov2","swav","byol","pirl","supervised r50",
                        # "supervised r18","supervised d121","mimic-chexpert lr=0.01",
                        # "mimic-chexpert lr=0.1","mimic-chexpert lr=1.0","mimic-cxr r18",
                        # "mimic-cxr d121"] 
                markers=['P', 'P', 'P','P', 'P', 'H', 'H', 'H', 'P', 'P', 'P', 'P', 'P']  
                colors = ['cornflowerblue', 'royalblue', 'lightskyblue', 'deepskyblue', 'steelblue',
                        'orangered', 'lightcoral', 'firebrick',
                        'limegreen', 'forestgreen', 'darkgreen', 'springgreen', 'seagreen']
                g = sns.lmplot(x='acc', y="invariance", hue="Models", data=new_df, fit_reg=False, markers=markers, palette=colors, scatter_kws={"s": 150})
                sns.regplot(x='acc', y="invariance", data=new_df, scatter=False, ax=g.axes[0, 0])
                plt.ylabel(f"{invariance_type.title()} Invariance")
                plt.xlabel(f"{transfer_setting.title()} Accuracy")
                plt.title(dset)
                #plt.show()
                save_results_to = f'acc_vs_invariance/{dset}/{transfer_setting}/'
                mkdir_p(save_results_to)
                plt.savefig(save_results_to + f'acc_vs_{invariance_type}_invariance.jpg', bbox_inches = "tight")
            except KeyError:
                print(f'No data for {invariance_type} invariance on {dset} dataset.')
    except KeyError:
        print(f'No data for {setting_dset}.')


plot_dset_acc_vs_invariance('few shot', 'chexpert')