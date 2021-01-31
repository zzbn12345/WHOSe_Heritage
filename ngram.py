# -*- coding: utf-8 -*-
"""GRU_sequence+attention.ipynb
# Classifying OUV using GRU sequence model + Attention

## Imports
"""

import sys
sys.executable

from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from scipy.special import softmax

import pickle
import matplotlib.pyplot as plt

import torch.autograd.profiler as profiler

import random

from sklearn.metrics import confusion_matrix

from scipy.special import softmax

import pickle
#import matplotlib.pyplot as plt

import torchtext
from torchtext.data import get_tokenizer
#tokenizer = get_tokenizer('spacy')

print("PyTorch version {}".format(torch.__version__))
print("GPU-enabled installation? {}".format(torch.cuda.is_available()))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

args = Namespace(
    # Data and Path information
    frequency_cutoff=1,
    model_state_file='model.pth',
    ouv_csv='Data/ouv_with_splits_full.csv',
    #ouv_csv='Data/all_with_splits_full.csv',
    prior_csv = 'Data/Coappearance_matrix.csv',
    save_dir='model_storage/ngram/',
    vectorizer_file='vectorizer.json',
    # Model hyper parameters
    ngrams=2,
    hidden_dim=200, 
    # Training hyper parameters
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.0002,
    l2 = 1e-5,
    dropout_p=0.5,
    k = 3,
    fuzzy = True,
    fuzzy_how = 'uni',
    fuzzy_lambda = 0.1,
    num_epochs=100,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
)
# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

classes = ['Criteria i', 'Criteria ii', 'Criteria iii', 'Criteria iv', 'Criteria v', 'Criteria vi', 
            'Criteria vii', 'Criteria viii', 'Criteria ix', 'Criteria x', 'Others']

"""## Data Vectorization Classes

### The Vocabulary
"""

class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token) 
        
        
    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx, 
                'add_unk': self._add_unk, 
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
    
    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary
        
        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

"""### The Vectorizer"""
def sparse_to_tensor(M):
    """
    input: M is Scipy sparse matrix
    output: pytorch sparse tensor in GPU 
    """
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    Ms = torch.sparse.FloatTensor(indices, values, shape)
    return Ms.to_dense().to(args.device)

class OuvVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, ouv_vocab, ngrams, vectorizer):
        """
        Args:
            review_vocab (Vocabulary): maps words to integers
        """
        self.ouv_vocab = ouv_vocab
        self.ngrams = ngrams
        self.vectorizer = vectorizer
        
    def vectorize(self, data):
        """Create a tf_idf vector for the ouv data
        
        Args:
            data (str): the ouv description data
            ngrams (int): the maximum ngram value
        Returns:
            tf_idf (np.ndarray): the tf-idf encoding 
        """
        data = [data]
        tf_idf = self.vectorizer.transform(data)
        
        return sparse_to_tensor(tf_idf)[0]

    @classmethod
    def from_dataframe(cls, ouv_df, ngrams, cutoff=5):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            ouv_df (pandas.DataFrame): the ouv dataset
            cutoff (int): the parameter for frequency-based filtering
            ngrams (int): the maximum ngram value
        Returns:
            an instance of the OuvVectorizer
        """
        ouv_vocab = Vocabulary(add_unk=True)
        corpus=[]
        
        # Add top words if count > provided count
        word_counts = Counter()
        for data in ouv_df.data:
            corpus.append(data)
            for word in ngrams_iterator(data.split(' '),ngrams=ngrams):
                if word not in string.punctuation:
                    word_counts[word] += 1
                     
        for word, count in word_counts.items():
            if count > cutoff:
                ouv_vocab.add_token(word)
        
        # Create keyword arguments to pass to the 'tf-idf' vectorizer.
        kwargs = {
                'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
                'dtype': 'int32',
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': TOKEN_MODE,  # Split text into word tokens.
                'min_df': MIN_DOCUMENT_FREQUENCY,
        }
        vectorizer = TfidfVectorizer(**kwargs)

        # Learn vocabulary from training texts and vectorize training texts.
        vectorizer.fit_transform(corpus).astype('float32')

        return cls(ouv_vocab, ngrams, vectorizer)

    @classmethod
    def from_serializable(cls, contents, ngrams):
        """Instantiate a OuvVectorizer from a serializable dictionary
        
        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the OuvVectorizer class
        """
        ouv_vocab = Vocabulary.from_serializable(contents['ouv_vocab'])
        
        return cls(ouv_vocab=ouv_vocab, ngrams=ngrams)

    def to_serializable(self):
        """Create the serializable dictionary for caching
        
        Returns:
            contents (dict): the serializable dictionary
        """
        return {'ouv_vocab': self.ouv_vocab.to_serializable()}

"""### The Dataset"""

class OuvDataset(Dataset):
    def __init__(self, ouv_df, vectorizer):
        """
        Args:
            ouv_df (pandas.DataFrame): the dataset
            vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """
        self.ouv_df = ouv_df
        self._vectorizer = vectorizer

        self.train_df = self.ouv_df[self.ouv_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.ouv_df[self.ouv_df.split=='dev']
        self.validation_size = len(self.val_df)

        self.test_df = self.ouv_df[self.ouv_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, ouv_csv, ngrams, cutoff):
        """Load dataset and make a new vectorizer from scratch
        
        Args:
            ouv_csv (str): location of the dataset
        Returns:
            an instance of OuvDataset
        """
        ouv_df = pd.read_csv(ouv_csv)
        train_ouv_df = ouv_df[ouv_df.split=='train']
        return cls(ouv_df, OuvVectorizer.from_dataframe(train_ouv_df,ngrams=ngrams, cutoff=cutoff))
    
    @classmethod
    def load_dataset_and_load_vectorizer(cls, ouv_csv, vectorizer_filepath, ngrams):
        """Load dataset and the corresponding vectorizer. 
        Used in the case in the vectorizer has been cached for re-use
        
        Args:
            ouv_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of OuvDataset
        """
        ouv_df = pd.read_csv(ouv_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath, ngrams=ngrams)
        return cls(ouv_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath,ngrams):
        """a static method for loading the vectorizer from file
        
        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of ReviewVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return OuvVectorizer.from_serializable(json.load(fp),ngrams=ngrams)

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json
        
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe 
        
        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and component for labels (y_target and y_fuzzy)
        """
        row = self._target_df.iloc[index]

        ouv_vector = \
            self._vectorizer.vectorize(row.data)

        true_label = \
            np.fromstring(row.true[1:-1],dtype=float, sep=' ')
        if len(true_label)==10:
            true_label = np.append(true_label,0.0)
        
        fuzzy_label = \
            np.fromstring(row.fuzzy[1:-1],dtype=float, sep=' ')

        return {'x_data': ouv_vector,
                'y_target': true_label,
                'y_fuzzy': fuzzy_label
               }

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size  
    
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

"""## The Model: GRU sequential Model"""

class MLPClassifier(nn.Module):
    
    def __init__(self, embedding_size, hidden_dim, num_classes, dropout_p, 
                 pretrained_embeddings=None, padding_idx=0):
        """
        Args:
            embedding_size (int): size of the embedding vectors
            num_embeddings (int): number of embedding vectors
            hidden_dim (int): the size of the hidden dimension
            num_classes (int): the number of classes in classification
            dropout_p (float): a dropout parameter 
            pretrained_embeddings (numpy.array): previously trained word embeddings
                default is None. If provided, 
            padding_idx (int): an index representing a null position
        """
        super(MLPClassifier, self).__init__()

        self._dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        # mlp classifier
        intermediate_vector = F.relu(self.dropout(self.fc1(x_in)))
        prediction_vector = self.fc2(intermediate_vector)
        
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector

"""## Training Routine

### Helper Functions
"""

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_k_acc_val': 0,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_1_acc': [],
            'train_k_acc': [],
            'train_k_jac': [],
            'val_loss': [],
            'val_1_acc': [],
            'val_k_acc': [],
            'val_k_jac': [],
            'test_loss': -1,
            'test_1_acc': -1,
            'test_k_acc':-1,
            'test_k_jac':-1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        acc_tm1, acc_t = train_state['val_k_acc'][-2:]

        # If accuracy worsened
        if acc_t <= train_state['early_stopping_best_k_acc_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model from sklearn
            if acc_t > train_state['early_stopping_best_k_acc_val']:
                train_state['early_stopping_best_k_acc_val'] = acc_t
                torch.save(model.state_dict(), train_state['model_filename'])
                
            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

"""### Evaluation Metrics"""

def compute_cross_entropy(y_pred, y_target):
    y_target = y_target.cpu().float()
    y_pred = y_pred.cpu().float()
    criterion = nn.BCEWithLogitsLoss()
    return criterion(y_target, y_pred)

def compute_1_accuracy(y_pred, y_target):
    y_target_indices = y_target.max(dim=1)[1]
    y_pred_indices = y_pred.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target_indices).sum().item()
    return n_correct / len(y_pred_indices) * 100

def compute_k_accuracy(y_pred, y_target, k=3):
    y_pred_indices = y_pred.topk(k, dim=1)[1]
    y_target_indices = y_target.max(dim=1)[1]
    n_correct = torch.tensor([y_pred_indices[i] in y_target_indices[i] for i in range(len(y_pred))]).sum().item()
    return n_correct / len(y_pred_indices) * 100

def compute_k_jaccard_index(y_pred, y_target, k=3):
    y_target_indices = y_target.topk(k, dim=1)[1]
    y_pred_indices = y_pred.max(dim=1)[1]
    jaccard = torch.tensor([len(np.intersect1d(y_target_indices[i], y_pred_indices[i]))/
                            len(np.union1d(y_target_indices[i], y_pred_indices[i]))
                            for i in range(len(y_pred))]).sum().item()
    return jaccard / len(y_pred_indices)

def compute_jaccard_index(y_pred, y_target, k=3, multilabel=False):
    
    threshold = 1.0/(k+1)
    threshold_2 = 0.5
    
    if multilabel:
        y_pred_indices = y_pred.gt(threshold_2)
    else:
        y_pred_indices = y_pred.gt(threshold)
    
    y_target_indices = y_target.gt(threshold)
        
    jaccard = ((y_target_indices*y_pred_indices).sum(axis=1)/((y_target_indices+y_pred_indices).sum(axis=1)+1e-8)).sum().item()
    return jaccard / len(y_pred_indices)

def softmax_sensitive(T):
    T = torch.exp(T) - 1 + 1e-9
    if len(T.shape)==1:
        return T/T.sum()
    return  T/(T.sum(axis=1).unsqueeze(1))

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = args.device
    return torch.from_numpy(df.values).float().to(device)

def get_prior():
    prior = pd.read_csv(args.prior_csv,sep=';',names=classes[:-1], skiprows=1)
    prior['Others'] = 1
    prior = prior.T
    prior['Others'] = 1
    prior = df_to_tensor(prior)
    return(prior)

def compute_fuzzy_label(y_target, y_fuzzy, fuzzy=False, how='uni', lbd=0):
    '''
    Using two sets of prediction labels and fuzziness parameters to compute the fuzzy label in the form as 
    a distribution over classes
    
    Args:
    y_target (torch.Tensor) of shape (n_batch, n_classes): the true label of the ouv description
    y_fuzzy (torch.Tensor) of shape (n_batch, n_classes): the fuzzy label of the ouv description
    fuzzy (bool): whether or not to turn on the fuzziness option
    how (string): the way fuzziness weights are used, one of the options in {'uni', 'prior'}
    lbd (float): the scaler applied to the fuzziness of the label
    
    Returns:
    A pytorch Tensor of shape (n_batch, n_classes): The processed label in the form of distribution that add to 1
    '''
    assert y_target.shape == y_fuzzy.shape, 'target labels must have the same size'
    assert how in {'uni', 'prior', 'origin'}, '''how must be one of the two options in {'uni', 'prior'}'''
    
    if not fuzzy:
        return softmax_sensitive(y_target)
    
    if how == 'uni':
        y_label = y_target + lbd * y_fuzzy
        return softmax_sensitive(y_label)
    
    ### TO DO ###
    elif how == 'prior':
        prior = get_prior()
        y_inter = torch.matmul(y_target.float(),prior)
        y_inter = y_inter/(y_inter.max(dim=1, keepdim=True)[0])
        y_label = y_target + lbd * y_fuzzy * y_inter
        return softmax_sensitive(y_label).to(args.device)
    
    else:
        y_label = y_target + lbd
        return softmax_sensitive(y_label)

"""### General Utilities"""

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def initialization():
    set_seed_everywhere(args.seed, args.cuda)
    if args.reload_from_files:
        # training from a checkpoint
        dataset = OuvDataset.load_dataset_and_load_vectorizer(args.ouv_csv, args.vectorizer_file)

    else:
        # create dataset and vectorizer
        dataset = OuvDataset.load_dataset_and_make_vectorizer(args.ouv_csv, cutoff=args.frequency_cutoff, ngrams = args.ngrams)
        dataset.save_vectorizer(args.vectorizer_file)    
    
    vectorizer = dataset.get_vectorizer()
    embedding_size = len(vectorizer.vectorizer.vocabulary_)
    classifier = MLPClassifier(embedding_size=embedding_size, 
                            hidden_dim=args.hidden_dim, 
                            num_classes=len(classes), 
                            dropout_p=args.dropout_p)
    return dataset, vectorizer, classifier

def training_loop(verbose=False):
    
    dataset,vectorizer,classifier = initialization()
    classifier = classifier.to(args.device)

    loss_func = cross_entropy
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                               mode='min', factor=0.5,
                                               patience=1)

    train_state = make_train_state(args)

    if verbose:
        epoch_bar = tqdm(desc='training routine', 
                              total=args.num_epochs)
        train_bar = tqdm(desc='split=train',
                              total=dataset.get_num_batches(args.batch_size), 
                              leave=True)
        val_bar = tqdm(desc='split=val',
                            total=dataset.get_num_batches(args.batch_size), 
                            leave=True)

    dataset.set_split('train')
    dataset.set_split('val')

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split('train')
            batch_generator = generate_batches(dataset, 
                                               batch_size=args.batch_size, 
                                               device=args.device)
            running_loss = 0.0
            running_1_acc = 0.0
            running_k_acc = 0.0
            running_k_jac = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):

                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. get the data compute fuzzy labels
                X = batch_dict['x_data']

                y_target = batch_dict['y_target']
                y_fuzzy = batch_dict['y_fuzzy']

                Y = compute_fuzzy_label(y_target, y_fuzzy, fuzzy= args.fuzzy, 
                                        how=args.fuzzy_how, lbd = args.fuzzy_lambda)

                # step 3. compute the output
                y_pred = classifier(X)

                # step 4. compute the loss
                loss = loss_func(y_pred, Y)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 5. use loss to produce gradients
                loss.backward()

                # step 6. use optimizer to take gradient step
                optimizer.step()

                # -----------------------------------------
                # compute the accuracies
                acc_1_t = compute_1_accuracy(y_pred, Y)
                acc_k_t = compute_k_accuracy(y_pred, Y, args.k)
                jac_k_t = compute_jaccard_index(y_pred, Y, len(classes))

                running_1_acc += (acc_1_t - running_1_acc) / (batch_index + 1)
                running_k_acc += (acc_k_t - running_k_acc) / (batch_index + 1)
                running_k_jac += (jac_k_t - running_k_jac) / (batch_index + 1)

                # update bar
                if verbose:
                    train_bar.set_postfix(loss=running_loss, 
                                      acc_1=running_1_acc,
                                      acc_k=running_k_acc,
                                      jac_k=running_k_jac,
                                      epoch=epoch_index)
                    train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_1_acc'].append(running_1_acc)
            train_state['train_k_acc'].append(running_k_acc)
            train_state['train_k_jac'].append(running_k_jac)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset, 
                                               batch_size=args.batch_size, 
                                               device=args.device)
            running_loss = 0.0
            running_1_acc = 0.0
            running_k_acc = 0.0
            running_k_jac = 0.0
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # step 2. get the data compute fuzzy labels
                X = batch_dict['x_data']

                y_target = batch_dict['y_target']
                y_fuzzy = batch_dict['y_fuzzy']

                Y = compute_fuzzy_label(y_target, y_fuzzy, fuzzy= args.fuzzy, 
                                        how=args.fuzzy_how, lbd = args.fuzzy_lambda)

                # step 3. compute the output
                with torch.no_grad():
                    y_pred = classifier(X)

                # step 4. compute the loss
                loss = loss_func(y_pred, Y)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # -----------------------------------------
                # compute the accuracies
                acc_1_t = compute_1_accuracy(y_pred, Y)
                acc_k_t = compute_k_accuracy(y_pred, Y, args.k)
                jac_k_t = compute_jaccard_index(y_pred, Y, len(classes))

                running_1_acc += (acc_1_t - running_1_acc) / (batch_index + 1)
                running_k_acc += (acc_k_t - running_k_acc) / (batch_index + 1)
                running_k_jac += (jac_k_t - running_k_jac) / (batch_index + 1)

                # update bar
                if verbose:
                    val_bar.set_postfix(loss=running_loss, 
                                    acc_1=running_1_acc,
                                    acc_k=running_k_acc,
                                    jac_k=running_k_jac,
                                    epoch=epoch_index)
                    val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_1_acc'].append(running_1_acc)
            train_state['val_k_acc'].append(running_k_acc)
            train_state['val_k_jac'].append(running_k_jac)

            train_state = update_train_state(args=args, model=classifier,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

            if verbose:
                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

    except KeyboardInterrupt:
        print("Exiting loop")
        pass
    
    return train_state

def update_best_config(current_best,train_state, key):
    val_k_acc = train_state['early_stopping_best_k_acc_val']
    if val_k_acc > current_best['k_acc']:
        current_best['k_acc'] = val_k_acc
        current_best['args'] = str(args)
        #current_best['state'] = [train_state]
        current_best['key'] = key

def Hypersearch(hyperdict, current_best):
    '''
    Perform a hyperparameter search using grid search and save into the hyperdict
    '''
    s_l2 = [0,1e-5,1e-4]
    s_dropout = [0.1,0.2,0.5]
    s_hidden_dim = [50, 100, 150, 200]
    s_batch_size = [64,128,256]
    
    search_bar = tqdm(desc='hyper_searching', 
                              total=len(s_l2)*len(s_dropout)*len(s_hidden_dim)*len(s_batch_size))
    
    i=0
    
    for l2 in s_l2:
        for dim in s_hidden_dim:
            for bs in s_batch_size:
                for dp in s_dropout:
                    args.l2 = l2
                    args.dropout = dp
                    args.hidden_dim = dim
                    args.batch_size = bs

                    key = (l2, dp, dim, bs)

                    if not key in hyperdict:
                        train_state = training_loop(verbose=True)
                        hyperdict[key] = train_state
                        update_best_config(current_best,train_state)

                    search_bar.set_postfix(best_k_acc = current_best['k_acc'],
                                           config = key)
                    search_bar.update()
                    i+=1

                    if i%100==0:
                        best_df = pd.DataFrame(current_best)
                        best_df.to_csv(args.save_dir+'best_config.csv')

    best_df = pd.DataFrame(current_best)
    best_df.to_csv(args.save_dir+'best_config.csv')

def fuzzy_exp(hyperdict,verbose):
    '''
    Perform a hyperparameter search using grid search and save into the hyperdict
    '''
    #s_fuzzy_how = ['uni','prior', 'origin']
    #s_fuzzy_lambda = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    s_fuzzy_how = ['uni']
    s_fuzzy_lambda = [0.1]
    
    search_bar = tqdm(desc='hyper_searching', 
                              total=len(s_fuzzy_lambda)*len(s_fuzzy_how))
    
    i=0
    
    for how in s_fuzzy_how:
        for lbd in s_fuzzy_lambda:
            args.fuzzy_how = how
            args.fuzzy_lambda = lbd
            
            key = (how, lbd)

            if not key in hyperdict:
                train_state = training_loop(verbose=verbose)
                hyperdict[key] = train_state
                
            search_bar.set_postfix(best_k_acc = hyperdict[key]['early_stopping_best_k_acc_val'],
                                   config = key)
            search_bar.update()
            i+=1
            with open(args.save_dir+'hyperdict_fuzzy.p', 'wb') as fp:
                 pickle.dump(hyperdict,fp, protocol=pickle.HIGHEST_PROTOCOL)

def ngrams_iterator(token_list, ngrams):
    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])
    for x in token_list:
        yield x
    for n in range(2, ngrams+1):
        for x in _get_ngrams(n):
            yield ' '.join(x)

def main():
    
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        
    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False
    else:
        torch.backends.cudnn.benchmark = True
        print('Using cudnn.benchmark.')

    print("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    #s_seed = [0,1,2,42,100,233,1024,1337,2333,4399]
    s_seed = [1337]

    for sd in s_seed:
        args.seed = sd
        args.save_dir = args.save_dir + '{}/'.format(sd)

        # Set seed for reproducibility
        set_seed_everywhere(args.seed, args.cuda)

        # handle dirs
        handle_dirs(args.save_dir)

        # Initialization
        if args.reload_from_files:
          #training from a checkpoint
            dataset = OuvDataset.load_dataset_and_load_vectorizer(args.ouv_csv, args.vectorizer_file)

        else:
           # create dataset and vectorizer
            dataset = OuvDataset.load_dataset_and_make_vectorizer(args.ouv_csv, cutoff=args.frequency_cutoff,ngrams = args.ngrams)
            dataset.save_vectorizer(args.vectorizer_file)    

        vectorizer = dataset.get_vectorizer()
        #set_seed_everywhere(args.seed, args.cuda)

        embedding_size = len(vectorizer.vectorizer.vocabulary_)
        
        classifier = MLPClassifier(embedding_size=embedding_size, 
                                hidden_dim=args.hidden_dim, 
                                num_classes=len(classes), 
                                dropout_p=args.dropout_p)

        # Train Model with Hyperparameter Search
        #hyperdict = {}
        #current_best = {}
        #current_best['loss'] = 1e10
        #current_best['1_acc'] = 0
        #current_best['k_acc'] = 0
        #current_best['k_jac'] = 0
        #current_best['args'] = None
        #current_best['state'] = None
        #current_best['key'] = None

        #Hypersearch(hyperdict, current_best,embeddings=embeddings,verbose=True)

        #with open(args.save_dir+'hyperdict.p', 'wb') as fp:
        #    pickle.dump(hyperdict,fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        #hid, l2, bs, bi, lr, dp = current_best['key']
        #args.hidden_dim = hid
        #args.l2 = l2
        #args.batch_size = bs
        #args.bi = bi
        #args.learning_rate = lr
        #args.dropout_p = dp

        hyperdict_fuzzy = {}

        fuzzy_exp(hyperdict = hyperdict_fuzzy, verbose=True)

        with open(args.save_dir+'hyperdict_fuzzy.p', 'wb') as fp:
            pickle.dump(hyperdict_fuzzy,fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
"""## END"""