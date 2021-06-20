import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def plot(pos_data, p_fakes, neg_data, n_fakes, gt_data, epoch):
    RS = 25111993
    pos_data = pos_data.tolist()
    p_fakes = p_fakes.tolist()

    neg_data = neg_data.tolist()
    n_fakes = n_fakes.tolist()

    gt_data = gt_data.tolist()

    for i in range(10):
        per = 30 * (i+1)
        lr = 200 * (i+1)



        ## Psotive real vs Posistive fake
        # Create features
        pos_data = pd.DataFrame(pos_data)
        p_fakes = pd.DataFrame(p_fakes)
        features = pd.concat([pos_data, p_fakes], axis=0)

        # Fit to TSNE
        digits_proj = TSNE(random_state=RS,  perplexity=per,  learning_rate=lr).fit_transform(features)
        x = digits_proj[:,0]
        y = digits_proj[:,1]

        x = pd.DataFrame(x,columns =['x'])
        y = pd.DataFrame(y,columns =['y'])

        # Create lable
        label_pos_data = ["Positive Data"] * len(pos_data)
        label_pfk_data = ["Positive Synthetic Data"] * len(p_fakes)

        label_pos_data = pd.DataFrame(label_pos_data, columns=['Class'])
        label_pfk_data = pd.DataFrame(label_pfk_data, columns=['Class'])
        label = pd.concat([label_pos_data, label_pfk_data], axis=0)

        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        label.reset_index(drop=True, inplace=True)

        #Create data
        data = pd.concat([x,y,label],axis=1)


        facet = sns.lmplot(data=data, x='x', y='y', hue='Class', fit_reg=False, legend=False, scatter_kws={"s": 1})

        #add a legend
        leg = facet.ax.legend(bbox_to_anchor=[1, 1],
                                 title="Class", fancybox=False)
        #change colors of labels
        customPalette = ['#0c0107', '#0c0107']
        for i, text in enumerate(leg.get_texts()):
            plt.setp(text, color = customPalette[i])

        #plt.show()
        epoch = str(epoch)
        per = str(per)
        lr = str(lr)
        plt.savefig('plots/New folder/positive_real_synthetic/Postive_Original_Synthetic_Data_Epoch_'+epoch+'Per = '+per+'Learning = '+lr+'.png', bbox_inches='tight')

        ###############################
        ###############################
        ## Negative real vs Negative fake
        ###############################
        ###############################



        neg_data = pd.DataFrame(neg_data)
        n_fakes = pd.DataFrame(n_fakes)
        features = pd.concat([neg_data, n_fakes], axis=0)

        # Fit to TSNE
        digits_proj = TSNE(random_state=RS).fit_transform(features)
        x = digits_proj[:,0]
        y = digits_proj[:,1]

        x = pd.DataFrame(x,columns =['x'])
        y = pd.DataFrame(y,columns =['y'])

        # Create lable
        label_neg_data = ["Negative Data"] * len(neg_data)
        label_nfk_data = ["Negative Synthetic Data"] * len(n_fakes)

        label_neg_data = pd.DataFrame(label_neg_data, columns=['Class'])
        label_nfk_data = pd.DataFrame(label_nfk_data, columns=['Class'])
        label = pd.concat([label_neg_data, label_nfk_data], axis=0)

        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        label.reset_index(drop=True, inplace=True)

        #Create data
        data = pd.concat([x,y,label],axis=1)


        facet = sns.lmplot(data=data, x='x', y='y', hue='Class', fit_reg=False, legend=False, scatter_kws={"s": 1})

        #add a legend
        leg = facet.ax.legend(bbox_to_anchor=[1, 1],
                                 title="Class", fancybox=False)
        #change colors of labels
        customPalette = ['#0c0107', '#0c0107']
        for i, text in enumerate(leg.get_texts()):
            plt.setp(text, color = customPalette[i])

        #plt.show()
        plt.savefig('plots/New folder/negative_real_synthetic/Negative_Original_Synthetic_Data_Epoch_'+epoch+'Per = '+str(per)+'Learning = '+str(lr)+'.png', bbox_inches='tight')

        ###############################
        ###############################
        ## Real vs Synthetic data
        ###############################
        ###############################


        gt_data = pd.DataFrame(gt_data)
        features = pd.concat([gt_data, p_fakes, n_fakes], axis=0)

        # Fit to TSNE
        digits_proj = TSNE(random_state=RS).fit_transform(features)
        x = digits_proj[:,0]
        y = digits_proj[:,1]

        x = pd.DataFrame(x,columns =['x'])
        y = pd.DataFrame(y,columns =['y'])

        # Create lable
        label_gt_data = ["Original Data"] * len(gt_data)
        label_fk_data = ["Synthetic Data"] * (len(pos_data)+len(neg_data))

        label_gt_data = pd.DataFrame(label_gt_data, columns=['Class'])
        label_fk_data = pd.DataFrame(label_fk_data, columns=['Class'])
        label = pd.concat([label_gt_data, label_fk_data], axis=0)

        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        label.reset_index(drop=True, inplace=True)

        #Create data
        data = pd.concat([x,y,label],axis=1)


        facet = sns.lmplot(data=data, x='x', y='y', hue='Class', fit_reg=False, legend=False, scatter_kws={"s": 1})

        #add a legend
        leg = facet.ax.legend(bbox_to_anchor=[1, 1],
                                 title="Class", fancybox=False)
        #change colors of labels
        customPalette = ['#0c0107', '#0c0107']
        for i, text in enumerate(leg.get_texts()):
            plt.setp(text, color = customPalette[i])

        #plt.show()
        plt.savefig('plots/New folder/real_synthetic/Original_Synthetic_Data_Epoch_'+epoch+'Per = '+str(per)+'Learning = '+str(lr)+'.png', bbox_inches='tight')
