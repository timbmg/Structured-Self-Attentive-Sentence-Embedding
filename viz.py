import argparse
import torch
from torch.utils.data import DataLoader
from dataset import YelpDataset
from selfattentive import SelfAttentive


def main(args):

    model = SelfAttentive.load(args.model)

    dataset = YelpDataset(args.data_dir)
    lengths = [args.num_datapoints, len(dataset)-args.num_datapoints]
    viz_set, _ = torch.utils.data.dataset.random_split(dataset, lengths=lengths)
    loader = DataLoader(dataset=viz_set, batch_size=args.num_datapoints, collate_fn=dataset.collate_fn, shuffle=False)

    batch = next(iter(loader))
    stars, sequences = batch['stars'] + 1, batch['sequences']

    _, _, _, attention = model(batch['sequences'], batch['lengths'], batch['stars'])
    attention = torch.sum(attention, 1)
    html = """
    <html>
    <body>
    """

    for bi in range(args.num_datapoints):

        norm_attention = ((attention[bi] - torch.min(attention[bi])) / (torch.max(attention[bi]) - torch.min(attention[bi]))).tolist()

        text = dataset.dictionary.decode(sequences[bi].tolist())

        html += """Stars: %i </br>"""%(stars[bi].item())

        for ti, ai in zip(text.split(), norm_attention):
            html += """<span style="background-color: #FF%s%s">%s </span>"""%(toHex(ai*255), toHex(ai*255), ti)

        html += """</br></br>"""

    html += """
    </body>
    </html>
    """

    with open('viz.html', 'w') as fout:
        fout.write(html)

def toHex(n):
    n = int(n)
    n = 255-n
    return hex(n)[2:].zfill(2).upper()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
        help="Directory of unzipped yelp dataset.")
    parser.add_argument('-n', '--num-datapoints', type=int, default=100,
        help="Number of examples to visualize.")
    parser.add_argument('-m', '--model', type=str, default='model.pt',
        help="Path to model to load.")

    args=parser.parse_args()

    main(args)