# A Structured Self-Attentive Sentence Embedding
Re-Implementation of [_A Structured Self-Attentive Sentence Embedding_](https://arxiv.org/pdf/1703.03130.pdf) by Lin et al., 2017.

# Results

| Set        | Loss     | Accuracy |
|:-----------|---------:|---------:|
| Training   | 1.700    | 66.70%   |
| Validation | 1.694    | 61.80%   |

Note that the above results were obtained after training for only 3 epochs. The training set contained 20000 examples and validation set 1000 examples. The training set size in the paper is much bigger.

## Data
The Yelp dataset can be download [here](http://pan.webis.de/clef16/pan16-web/author-profiling.html). 
After downloading, the file only has to be unzipped.

## Training
You can run the training procedure with the default settings with the following command:  
`python3 train.py --data-dir <dir of unzipped yelp data>`

For more information about training settings run:  
`python3 train.py --help` 

## Visualization
Once the model is trained, the attention pattern can be visualized as done in the paper. 
The following python script will create an HTML file with the reviews and respective attention pattern.  
`python3 viz.py`

An Example of 5 reviews with different ratings is given below:  
![](imgs/viz2.png)

## Differences with the paper
- Adam instead of SGD
- No gradient clipping
- No dropout