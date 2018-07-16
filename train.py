import os
import torch
import argparse
from torch.utils.data import DataLoader
from selfattentive import SelfAttentive
from dataset import YelpDataset

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets & Dataloader
    dataset = YelpDataset(args.data_dir, args.min_occurance, size=args.train_size+args.valid_size)
    train_set, valid_set = torch.utils.data.dataset.random_split(dataset, lengths=[args.train_size, args.valid_size])
    valid_set.dataset.save(os.path.join(args.data_dir, 'valid.json'))
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    # Model
    model = SelfAttentive(
            num_embeddings=len(dataset.dictionary), 
            num_embedding_hidden=args.num_embedding_hidden, 
            num_encoder_hidden=args.num_encoder_hidden,
            num_attention_hidden=args.num_attention_hidden, 
            num_classifier_hidden=args.num_classifier_hidden,
            num_hops=args.num_hops, 
            rnn_cell='lstm', 
            bidirectional=True
            )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training
    print("Training...")
    best_valid_loss = 1e99
    for epoch in range(1, args.epochs+1):
        train_loss, train_accuracy = evaluate_epoch(model=model, data_loader=train_loader, optimizer=optimizer)
        valid_loss, valid_accuracy = evaluate_epoch(model=model, data_loader=valid_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save(args.model_file_name)
        
        print("Epoch %02d/%02d Train Loss %5.3f Train Acc %5.3f Valid Loss %5.3f Valid Acc %5.3f"
            %(epoch, args.epochs, train_loss, train_accuracy*100, valid_loss, valid_accuracy*100))


def evaluate_epoch(model, data_loader, optimizer=None):
    
    epoch_loss, epoch_acc = 0, 0

    if optimizer is not None:
        torch.enable_grad()
        model.train()
    else:
        torch.no_grad()
        model.eval()
    
    for batch in data_loader:
        loss, penalization_term, accuracy = model(batch['sequences'], batch['lengths'], batch['stars'])
        loss += penalization_term

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += accuracy

    return epoch_loss/len(data_loader), epoch_acc/len(data_loader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
        help="Directory of unzipped yelp dataset.")
    parser.add_argument('-mo', '--min-occurance', type=int, default=3,
        help="The minimum number of occurances of a word.")
    parser.add_argument('--train-size', type=int, default=20000,
        help="Number of datapoints in in the training set.")
    parser.add_argument('--valid-size', type=int, default=1000,
        help="Number of datapoints in the validation set.")
    parser.add_argument('-m', '--model-file-name', type=str, default='model.pt',
        help="File to save parameters to.")

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch-size', type=int, default=16)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)

    parser.add_argument('-emb', '--num-embedding-hidden', type=int, default=100,
        help="Number of dimensions of the embedding layer.")
    parser.add_argument('-hid', '--num-encoder-hidden', type=int, default=300,
        help="Number of dimensions of the hidden state of the Encoder.")
    parser.add_argument('-att', '--num-attention-hidden', type=int, default=350,
        help="Number of dimenions of the hidden layer of the attention mechanism.")
    parser.add_argument('-cls', '--num-classifier-hidden', type=int, default=2000,
        help="Number of dimensions of the hidden layer of the classifier.")
    parser.add_argument('-hop', '--num-hops', type=int, default=30,
        help="Number of attention hops.")

    args = parser.parse_args()
    
    main(args)