import argparse
import torch
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from dataset import YelpDataset
from selfattentive import SelfAttentive

torch.no_grad()

def main(args):

    print("Loading Model...")
    model = SelfAttentive.load(args.model)
    model.eval()

    dataset = YelpDataset(args.data_dir, load_from=args.validation_set)
    loader = DataLoader(dataset=dataset, batch_size=args.num_datapoints, collate_fn=dataset.collate_fn, shuffle=False)

    if args.html:  
        print("Creating HTML...")  
        batch = next(iter(loader))
        stars, sequences = batch['stars'] + 1, batch['sequences']

        _ = model(batch['sequences'], batch['lengths'], batch['stars'])
        attention = torch.sum(model.attention_weights, 1)
        html = str()

        for bi in range(args.num_datapoints):
            norm_attention = ((attention[bi] - torch.min(attention[bi])) / (torch.max(attention[bi]) - torch.min(attention[bi]))).tolist()
            text = dataset.dictionary.decode(sequences[bi].tolist())

            html += """Stars: %i </br>"""%(stars[bi].item())
            for ti, ai in zip(text.split(), norm_attention):
                html += """<span style="background-color: #FF%s%s">%s </span>"""%(toHex(ai*255), toHex(ai*255), ti)

            html += """</br></br>"""

        with open('viz.html', 'w') as fout:
            fout.write(html)
    
    if args.tsne or args.cm:
        loader = DataLoader(dataset=dataset, batch_size=32, collate_fn=dataset.collate_fn, shuffle=False)
        sentence_embeddings = list()
        stars = list()
        predictions = list()
        n = 100 # number of datapoints for TSNE
        print("Predicting...")
        for i, batch in enumerate(loader):
            _ = model(batch['sequences'], batch['lengths'], batch['stars'])
            stars.append(batch['stars'])
            predictions.append(model.predictions.topk(1)[1].squeeze(1))
            if i*32 <= n:
                sentence_embeddings.append(model.sentence_embedding)

    # Confusion Matrix
    stars = torch.cat(stars, dim=0).detach().numpy()
    if args.cm:
        print("Creating Confusion Matrix...")
        predictions = torch.cat(predictions).detach().numpy()
        cm = confusion_matrix(predictions, stars)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(
            cm, index=[str(i) for i in range(1, 5+1)], columns=[str(i) for i in range(1, 5+1)], 
        )
        figsize = (10,7)
        fontsize=1
        fig = plt.figure(figsize=figsize)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="1.2f")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')#, fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')#, ha='right', fontsize=fontsize)
        plt.ylabel('True Stars')
        plt.xlabel('Predicted Stars')
        plt.savefig('cm.png', dpi=300)
        plt.show()

    # TSNE
    if args.tsne:
        print("Creating TSNE...")
        sentence_embeddings = (torch.cat(sentence_embeddings, dim=0)/30).detach().numpy()
        feat_cols = [str(i) for i in range(sentence_embeddings.shape[1])]
        df = pd.DataFrame(sentence_embeddings, columns=feat_cols)
        df['stars'] = stars[:df.shape[0]] + 1

        tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(df[feat_cols].values)

        df_tsne = df.copy()
        df_tsne['x-tsne'] = tsne_results[:,0]
        df_tsne['y-tsne'] = tsne_results[:,1]

        sns.set(rc={'figure.figsize':(10,20), 'figure.dpi':200})
        sns.lmplot(x="x-tsne", y="y-tsne", data=df_tsne, fit_reg=False, hue='stars', legend=False)
        plt.legend(loc='best')
        plt.savefig('tsne.png', dpi=300)
        plt.show()


def toHex(n):
    n = int(n)
    n = 255-n
    return hex(n)[2:].zfill(2).upper()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
        help="Directory of unzipped yelp dataset.")
    parser.add_argument('-val', '--validation-set', type=str,
        help="File path to validation set JSON.")
    parser.add_argument('-n', '--num-datapoints', type=int, default=100,
        help="Number of examples to visualize.")
    parser.add_argument('-m', '--model', type=str, default='model.pt',
        help="Path to model to load.")

    parser.add_argument('-html', '--html', action='store_true')
    parser.add_argument('-tsne', '--tsne', action='store_true')
    parser.add_argument('-cm', '--cm', action='store_true')

    args=parser.parse_args()

    main(args)