import pandas as pd
import numpy as np
import torch
import sklearn
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from nltk import word_tokenize



text_datasets = set(["imdb", "twitter"])

class XYWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __iter__(self):
        for batch in self.dataset:
            x = batch.text
            y = batch.label

            yield (x, y)
    
    def __len__(self):
        return len(self.dataset)
    
def get_text_iterator(train_paths, test_paths, args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    TEXT = Field(sequential=True, tokenize=word_tokenize, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)
    
    datafields = [("text", TEXT),
                 ("label", LABEL)]

    trains = []
    for train_path in train_paths:
        trains.append( 
            TabularDataset(
                path=train_path, 
                format='csv',
                skip_header=True, 
                fields=datafields)
            )

    tests = []
    for test_path in test_paths:
        tests.append( 
            TabularDataset(
                path=test_path, 
                format='csv',
                skip_header=True, 
                fields=datafields)
            )
    
    TEXT.build_vocab(*trains, vectors="glove.6B.100d")
    vocab = TEXT.vocab

    train_iters = []
    test_iters = []

    for train in trains:
        train_iter = BucketIterator(train, batch_size=64, device=device,
                sort_key=lambda x: len(x.text), 
                sort_within_batch=False,
                repeat=False 
        )
        train_iters.append(XYWrapper(train_iter))

    for test in tests:
        test_iter = Iterator(test, batch_size=64, device=device, sort=False, sort_within_batch=False, repeat=False)
        test_iters.append(XYWrapper(test_iter))
    
    
    return train_iters, test_iters, vocab


def get_path(name):

    if name == "twitter":
        train_path = "data/twitter_train.csv"
        test_path = "data/twitter_test.csv"
        return train_path, test_path
    
    if name == "imdb":
        train_path = "data/imdb_train.csv"
        test_path = "data/imdb_test.csv"
        return train_path, test_path

def get_text_dataloader(names, args):

    train_paths = []
    test_paths = []
    for name in names:
        train_path, test_path = get_path(name)
        train_paths.append(train_path)
        test_paths.append(test_path)

    return get_text_iterator(train_paths, test_paths, args)


