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

from torch.autograd import Variable

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
    save_dir='model_storage/attn/',
    vectorizer_file='vectorizer.json',
    # Model hyper parameters
    glove_filepath='Data/glove/glove.6B.300d.txt', 
    use_glove=True,
    freeze = True,
    embedding_size=300, 
    hidden_dim=128, 
    bi = False,
    # Training hyper parameters
    batch_size=256,
    early_stopping_criteria=5,
    learning_rate=0.001,
    l2=1e-5,
    dropout_p=0.1,
    k = 3,
    fuzzy = True,
    fuzzy_how = 'uni',
    fuzzy_lambda = 0.2,
    num_epochs=100,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
)

classes = ['Criteria i', 'Criteria ii', 'Criteria iii', 'Criteria iv', 'Criteria v', 'Criteria vi', 
            'Criteria vii', 'Criteria viii', 'Criteria ix', 'Criteria x', 'Others']

"""## Data Vectorization Classes

### The Vocabulary
"""

class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None):
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
                
    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx}

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

class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents
    
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

"""### The Vectorizer"""

class OuvVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, ouv_vocab):
        """
        Args:
            review_vocab (Vocabulary): maps words to integers
        """
        self.ouv_vocab = ouv_vocab
        
    def vectorize(self, data, vector_length = -1):
        """Create a collapsed one-hit vector for the ouv data
        
        Args:
            data (str): the ouv description data
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vectorized data (np.ndarray)
        """
        indices = []
        indices.extend(self.ouv_vocab.lookup_token(token) for token in data.split(' '))
        
        if vector_length < 0:
            vector_length = len(indices)
            
        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.ouv_vocab.mask_index
        
        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, ouv_df, cutoff=5):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            ouv_df (pandas.DataFrame): the ouv dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the OuvVectorizer
        """
        
        # Add top words if count > provided count
        word_counts = Counter()
        for data in ouv_df.data:
            for word in data.split(' '):
                if word not in string.punctuation:
                    word_counts[word] += 1
        
        ouv_vocab = SequenceVocabulary()
        for word, count in word_counts.items():
            if count > cutoff:
                ouv_vocab.add_token(word)

        return cls(ouv_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a OuvVectorizer from a serializable dictionary
        
        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the OuvVectorizer class
        """
        ouv_vocab = SequenceVocabulary.from_serializable(contents['ouv_vocab'])
        
        return cls(ouv_vocab=ouv_vocab)

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
        
        # +0 if not using begin_seq and end seq, +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, ouv_df.data)) + 0

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
    def load_dataset_and_make_vectorizer(cls, ouv_csv, cutoff):
        """Load dataset and make a new vectorizer from scratch
        
        Args:
            ouv_csv (str): location of the dataset
            cutoff (int): the boundary to set the words into unknown
        Returns:
            an instance of OuvDataset
        """
        ouv_df = pd.read_csv(ouv_csv)
        train_ouv_df = ouv_df[ouv_df.split=='train']
        return cls(ouv_df, OuvVectorizer.from_dataframe(train_ouv_df, cutoff=cutoff))
    
    @classmethod
    def load_dataset_and_load_vectorizer(cls, ouv_csv, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer. 
        Used in the case in the vectorizer has been cached for re-use
        
        Args:
            ouv_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of OuvDataset
        """
        ouv_df = pd.read_csv(ouv_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(ouv_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file
        
        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of ReviewVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return OuvVectorizer.from_serializable(json.load(fp))

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

        ouv_vector, vec_length = \
            self._vectorizer.vectorize(row.data, self._max_seq_length)

        true_label = \
            np.fromstring(row.true[1:-1],dtype=float, sep=' ')
        
        if len(true_label)==10:
            true_label = np.append(true_label,0.0)
        
        fuzzy_label = \
            np.fromstring(row.fuzzy[1:-1],dtype=float, sep=' ')

        return {'x_data': ouv_vector,
                'y_target': true_label,
                'y_fuzzy': fuzzy_label,
                'x_length': vec_length
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

class AttnGRUClassifier(nn.Module):
    
    def __init__(self, batch_size, embedding_size, num_embeddings, hidden_dim, num_classes, dropout_p,
                 batch_first=True, pretrained_embeddings=None, padding_idx=0, bi = False, freeze=True):
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
        super(AttnGRUClassifier, self).__init__()

        if pretrained_embeddings is None:

            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)        
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding.from_pretrained(pretrained_embeddings,
                                    padding_idx=padding_idx,
                                    freeze=True)
        
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout_p)
        self.padding_idx = padding_idx
        self.hidden_dim = hidden_dim
        self.bi = bi
        
        self.word_gru = nn.GRU(input_size = embedding_size, hidden_size=hidden_dim, batch_first=False,
                         num_layers=1, bidirectional = bi)
        
        self.word_bias = nn.Linear((bi+1)*hidden_dim, (bi+1)*hidden_dim)
        self.word_n_bias = nn.Linear((bi+1)*hidden_dim, 1, bias=False)
        
        self.fc = nn.Linear(hidden_dim * (bi+1), num_classes)

    def forward(self, x_in, state_word, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        # embed and permute so features are channels
        x_embedded = self.emb(x_in)
        padding_mask = x_in.eq(self.padding_idx)
        length = (~padding_mask).sum(dim=1)
        
        if self.batch_first:
            x_embedded = x_embedded.transpose(1,0)
        
        # rnn units
        output_word, state_word = self.word_gru(x_embedded, state_word)
        #(T,B,C)
        word_squish = self.word_bias(output_word.view(-1,output_word.shape[-1]))
        #(B*T,C)
        word_squish = torch.tanh(word_squish)
        word_attn = self.word_n_bias(word_squish).view(output_word.shape[0],output_word.shape[1],1)
        word_attn[padding_mask.transpose(0,1)]= -np.inf
        #(B*T,1)->(T,B,1)
        
        word_attn_norm = F.softmax(word_attn, dim=0).transpose(0,1)
        #(B,T,1)
        
        word_attn_vec = torch.bmm(output_word.permute(1,2,0),word_attn_norm)
        #(B,C,T)*(B,T,1)->(B,C,1)
        
        y_out = self.fc(self.dropout(word_attn_vec.squeeze()))
        
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
            
        return y_out, state_word, word_attn_norm.squeeze()
    
    def init_hidden(self):
        return Variable(torch.zeros((self.bi+1), self.batch_size, self.hidden_dim))

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

def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings 
    
    Args:
        glove_filepath (str): path to the glove embeddings file 
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ") # each line: word num1 num2 ...
            word_to_index[line[0]] = index # word = line[0] 
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)

def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.
    
    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    
    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings

def initialization(embeddings):
    set_seed_everywhere(args.seed, args.cuda)
    if args.reload_from_files:
        # training from a checkpoint
        dataset = OuvDataset.load_dataset_and_load_vectorizer(args.ouv_csv, args.vectorizer_file)

    else:
        # create dataset and vectorizer
        dataset = OuvDataset.load_dataset_and_make_vectorizer(args.ouv_csv, cutoff=args.frequency_cutoff)
        dataset.save_vectorizer(args.vectorizer_file)    
    
    vectorizer = dataset.get_vectorizer()
    classifier = AttnGRUClassifier(batch_size=args.batch_size,
                                   embedding_size=args.embedding_size, 
                                num_embeddings=len(vectorizer.ouv_vocab),
                                hidden_dim=args.hidden_dim, 
                                num_classes=len(classes), 
                                dropout_p=args.dropout_p,
                                pretrained_embeddings=embeddings,
                                padding_idx=0,
                                bi=args.bi, freeze=args.freeze)
    return dataset, vectorizer, classifier

def training_loop(embeddings, verbose=False):
    
    dataset,vectorizer,classifier = initialization(embeddings=embeddings)
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
                state_word = classifier.init_hidden().to(args.device)
            
                optimizer.zero_grad()

                # step 2. get the data compute fuzzy labels
                X = batch_dict['x_data']

                y_target = batch_dict['y_target']
                y_fuzzy = batch_dict['y_fuzzy']

                Y = compute_fuzzy_label(y_target, y_fuzzy, fuzzy= args.fuzzy, 
                                        how=args.fuzzy_how, lbd = args.fuzzy_lambda)

                # step 3. compute the output
                y_pred, state_word, _ = classifier(X, state_word)

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
                state_word = classifier.init_hidden().to(args.device)
            
                X = batch_dict['x_data']

                y_target = batch_dict['y_target']
                y_fuzzy = batch_dict['y_fuzzy']

                Y = compute_fuzzy_label(y_target, y_fuzzy, fuzzy= args.fuzzy, 
                                        how=args.fuzzy_how, lbd = args.fuzzy_lambda)

                # step 3. compute the output
                with torch.no_grad():
                    y_pred, state_word, _ = classifier(X, state_word)

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

def Hypersearch(hyperdict, current_best, embeddings, verbose):
    '''
    Perform a hyperparameter search using grid search and save into the hyperdict
    '''
    s_hid = [64, 128, 256]
    s_l2 = [0, 1e-5, 1e-4]
    s_batch_size = [64,128,256]
    s_bi = [True, False]
    s_lr = [0.001, 0.0005, 0.0002]
    s_dp = [0,0.1,0.2,0.5]
    
    search_bar = tqdm(desc='hyper_searching', 
                              total=len(s_hid)*len(s_l2)*len(s_batch_size)*len(s_bi)*len(s_lr)*len(s_dp))
    
    i=0
    
    for hid in s_hid:
        for l2 in s_l2:
            for bs in s_batch_size:
                for bi in s_bi:
                    for lr in s_lr:
                        for dp in s_dp:
                              
                            
                            args.hidden_dim = hid
                            args.l2 = l2
                            args.batch_size = bs
                            args.bi = bi
                            args.learning_rate = lr
                            args.dropout_p = dp

                            key = (hid, l2, bs, bi, lr, dp)

                            if not key in hyperdict:
                                train_state = training_loop(embeddings = embeddings, verbose=verbose)
                                hyperdict[key] = train_state
                                update_best_config(current_best,train_state, key)

                            search_bar.set_postfix(best_k_acc = current_best['k_acc'],
                                                config = current_best['key'])
                            search_bar.update()
                            
                            if i%20==0:
                                best_df = pd.DataFrame(current_best)
                                best_df.to_csv(args.save_dir+'best_config.csv')

                                with open(args.save_dir+'hyperdict.p', 'wb') as fp:
                                    pickle.dump(hyperdict,fp, protocol=pickle.HIGHEST_PROTOCOL)
                            i+=1
                                
    best_df = pd.DataFrame(current_best)
    best_df.to_csv(args.save_dir+'best_config.csv')

def fuzzy_exp(embeddings, hyperdict,verbose):
    '''
    Perform a hyperparameter search using grid search and save into the hyperdict
    '''
    #s_fuzzy_how = ['uni','prior', 'origin']
    #s_fuzzy_lambda = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    s_fuzzy_how = ['uni']
    s_fuzzy_lambda = [0.2]
    
    search_bar = tqdm(desc='hyper_searching', 
                              total=len(s_fuzzy_lambda)*len(s_fuzzy_how))
    
    i=0
    
    for how in s_fuzzy_how:
        for lbd in s_fuzzy_lambda:
            args.fuzzy_how = how
            args.fuzzy_lambda = lbd
            
            key = (how, lbd)

            if not key in hyperdict:
                train_state = training_loop(embeddings = embeddings, verbose=verbose)
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

def get_ngram_vocab():
    ouv_df = pd.read_csv(args.ouv_csv)
    word_counts = Counter()
    for data in ouv_df.data:
        token_list = data.split(' ')
        for word in ngrams_iterator(token_list, 3):
            temp = 0
            for element in word:
                if element in string.punctuation:
                    temp = 1
                    break
            if temp==0:
                word_counts[word] += 1

    vocab = [word for word, count in word_counts.items() if count>15 and count<600]
    return(vocab)

def infer_tokens_importance(vocab, classifier, vectorizer, classes, k=50):
    """Predict the rating of a review
    
    Args:
        vocab (list of str): the whole vocabulary
        classifier (ReviewClassifier): the trained model
        vectorizer (ReviewVectorizer): the corresponding vectorizer
        classes (list of str): The name of the ouv classes
        k (int): show the largest k prediction, default to 1
    """
    vectorized_token = []    
    for token in vocab:
        vectorized_token.append(torch.tensor(vectorizer.vectorize(token, vector_length=dataset._max_seq_length)[0]))
        
    X = torch.stack(vectorized_token)
    result = classifier(X, apply_softmax=True)
    
    vocab_id = result[1:].topk(k, dim=0)[1]
    vocab_weight = result[1:].topk(k, dim=0)[0]
    return vocab_id, vocab_weight

def make_top_k_DataFrame(vocab, classifier, vectorizer, classes, k=10):
    
    vocab_id = infer_tokens_importance(vocab, classifier, vectorizer, classes, k)[0]
    df = pd.DataFrame(columns = classes)
    for i in range(len(classes)):
        
        indices = vocab_id[:,i].tolist()
        words = pd.Series([vocab[j] for j in indices])
        df[classes[i]] = words
    return df

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

        set_seed_everywhere(args.seed, args.cuda)

        # handle dirs
        handle_dirs(args.save_dir)

        # Initialization
        if args.reload_from_files:
            #training from a checkpoint
            dataset = OuvDataset.load_dataset_and_load_vectorizer(args.ouv_csv, args.vectorizer_file)

        else:
            # create dataset and vectorizer
            dataset = OuvDataset.load_dataset_and_make_vectorizer(args.ouv_csv, cutoff=args.frequency_cutoff)
            dataset.save_vectorizer(args.vectorizer_file)    

        vectorizer = dataset.get_vectorizer()
        #set_seed_everywhere(args.seed, args.cuda)

        # Use GloVe or randomly initialized embeddings
        if args.use_glove:
            words = vectorizer.ouv_vocab._token_to_idx.keys()
            embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath, 
                                            words=words)
            print("Using pre-trained embeddings")
        else:
            print("Not using pre-trained embeddings")
            embeddings = None
        
        classifier = AttnGRUClassifier(batch_size=args.batch_size,
                                embedding_size=args.embedding_size,
                                num_embeddings=len(vectorizer.ouv_vocab),
                                hidden_dim=args.hidden_dim, 
                                num_classes=len(classes), 
                                dropout_p=args.dropout_p,
                                pretrained_embeddings=embeddings,
                                padding_idx=0,
                                bi = args.bi,
                                freeze = args.freeze)

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

        fuzzy_exp(hyperdict = hyperdict_fuzzy, embeddings=embeddings,verbose=True)

        with open(args.save_dir+'hyperdict_fuzzy.p', 'wb') as fp:
            pickle.dump(hyperdict_fuzzy,fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
"""## END"""