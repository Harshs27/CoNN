# importing modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
try:
   import cPickle as pickle
except:
   import pickle
import sys, os, re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from glob import glob
import string
from gensim import corpora, models, similarities
import math
import matplotlib.pyplot as plt
import time
from six import PY3, iteritems, iterkeys, itervalues, string_types
import itertools
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import svm, metrics

from preprocess import prepare_data

import argparse

parser = argparse.ArgumentParser(description='Running Cooperative Neural Network for document classification')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--USE_CUDA', type=str2bool, nargs='?', default=True, 
                    help='use GPUs if available, uses all available gpu by default::: If you want to run on specific GPUs, use CUDA_VISIBLE_DEVICES')

parser.add_argument('--PREP_DATA', type=str2bool, nargs='?', default=False, 
                    help='preprocess the data, the script saves the processed data for later reuse')

parser.add_argument('--DO_CV', type=str2bool, nargs='?', default=True, 
                    help='do cross validation')

parser.add_argument('--K_FOLD', type=int, default=5,
                    help='K fold validation')

parser.add_argument('--NUM_BATCHES', type=int, default=1600,
                    help='Number of batches for training')

parser.add_argument('--BATCH_SIZE', type=int, default=100,
                    help='Batch size')

parser.add_argument('--DROPOUT', type=float, default=0.01,
                    help='dropout for regularization')

parser.add_argument('--IMBALANCE_HANDLING', type=str2bool, nargs='?', default=False, 
                    help='apply cost-sensitive learning to the loss function')

parser.add_argument('--EVALUATE_TEST', type=str2bool, nargs='?', default=True, 
                    help='Report the results on validation dataset at EVALUATE_EVERY iterations')

parser.add_argument('--EVALUATE_EVERY', type=int, default=500,
                    help='evaluate at intervals of EVALUATE_EVERY epochs')

parser.add_argument('--hilbert_DIM', type=int, default=20,
                    help='The size of the hilbert space embedding vector')

parser.add_argument('--WORD2VEC', type=int, default=20,
                    help='the size of word2vec embeddings')

parser.add_argument('--num_CLASS', type=int, default=1,
                    help='number of classes for classification: 1 = binary classification')

parser.add_argument('--ITERATIONS', type=int, default=2,
                    help='the number of iterations of the CoNN algorithm1 to obtain embeddings')

parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')

parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use ')

args = parser.parse_args()



print('All modules imported')

USE_CUDA = args.USE_CUDA  
PREP_DATA =  args.PREP_DATA 
MODEL_NAME = 'CoNN'
K_FOLD = args.K_FOLD 
DROPOUT = args.DROPOUT
NUM_BATCHES = args.NUM_BATCHES 
BATCH_SIZE = args.BATCH_SIZE
DO_CV = args.DO_CV 
EVALUATE_TEST = args.EVALUATE_TEST 
EVALUATE_EVERY = args.EVALUATE_EVERY 
IMBALANCE_HANDLING = args.IMBALANCE_HANDLING 

hilbert_DIM = args.hilbert_DIM 
WORD2VEC = args.WORD2VEC 
num_CLASS = args.num_CLASS   
ITERATIONS = args.ITERATIONS 


def get_auc_plot(y, scores):
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    for i in range(len(fpr)):
        if fpr[i] > 0.01:
            break
    return roc_auc, tpr[i], fpr[i]

def get_auc(y, scores):
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)

    for i in range(len(fpr)):
        if fpr[i] > 0.01:
            break
    return roc_auc#, tpr[i], fpr[i]

with open('stopwords', encoding='utf-8', errors='ignore', mode='r') as myfile:
    extra_stopwords = myfile.read().replace('\n', ' ').split(' ')[:-1]
stop = set(stopwords.words('english'))
# update more stopwords
stop.update(extra_stopwords)
#stop.update(['would', 'like', 'know', 'also', 'may', 'use', 'dont', 'get', 'com', 'write', 'want', 'edu', 'articl', 'article'])
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop and not i.isdigit()])
    new_stop_free = " ".join(re.findall(r"[\w']+", stop_free))
    punc_free = ''.join(ch for ch in new_stop_free if ch not in exclude)
    stop_punc_free = " ".join([i for i in punc_free.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_punc_free.split())
    words = []
    for word in normalized.split():
        if word in ['oed', 'aing']:
            continue
        else:
            stemmed_word = stemmer.stem(word)
            if(len(stemmed_word)>2):
                words.append(stemmed_word)
    stemmed = " ".join(words)
    stemmed_stop_free = " ".join([i for i in stemmed.split() if i not in stop])
    return stemmed_stop_free

def doc2idx(dictionary, document, unknown_word_index=-1):
        if isinstance(document, string_types):
            raise TypeError("doc2idx expects an array of unicode tokens on input, not a single string")

        document = [word if isinstance(word, str) else unicode(word, 'utf-8') for word in document]
        return [dictionary.token2id.get(word, unknown_word_index) for word in document]


def prepare_data(basefile):
    data_cleaned = [] # format: <label, text>
    count = 0
    for path in sorted(glob(basefile+'/*')):
        category_type = os.path.basename(path)#path.split("/")[-1]
        print('category_type', category_type)
        for files in sorted(glob(basefile+'/'+category_type+'/processed.review')):#contains 342k reviews
            print('files', files)
            with open(files, encoding='utf-8', errors='ignore', mode='r') as myfile:
                file_data = myfile.readlines()#.replace('\n', ' ')
            file_data = [x.strip().split(' ') for x in file_data] # converting file data to list of lines
            for line in file_data:
                sentence = []
                label = 0 if line[-1].split(':')[-1] == 'negative' else 1 # negative label=0
                #print(line[0:-1])
                line_data = [w.split(':') for w in line[0:-1]]
#                print(line_data)
                for pair in line_data:
                    w  = ' '.join(pair[0].split('_'))# word
                    wc = int(pair[1])#word count
                    for c in range(wc): # count in wc
                        sentence.append(w)
                sentence = ' '.join(sentence)
                data_cleaned.append([label, clean(sentence).split()])
                count += 1
#                print(complete_data)
    print('total review texts ', count)
    return data_cleaned

def get_CV_data(data_cleaned, train_index, test_index, TEST_FLAG=0, dictionary='dummy_dictionary'):       
    print('Getting test and train datasets')
    count = 0
    current_cat = 0
    train_data = []
    test_data = []

    for idx in train_index:
        train_data.append(data_cleaned[idx])
    test_data = [data_cleaned[idx] for idx in test_index]

    print('# of train data', len(train_data))
    print('# of test data', len(test_data))
    
    # creating the dictionary from the train_data
    dict_data = [d[1] for d in train_data] # extracting only the text 
    # creating a dictionary, where every unique term is associated with an index
    if TEST_FLAG == 0:
        dictionary = corpora.Dictionary(dict_data)
    #else: # use the already trained dictionary
        
    # converting dataset into X and y
    train_cleaned_data = []
    train_labels = []
    train_num_words = []
    for d in train_data:
        only_known_words = list(filter((-1).__ne__, doc2idx(dictionary, document=d[1])))# should not include unknown words
        temp_len = len(only_known_words)
        if temp_len == 0:
            continue
        train_num_words.append(temp_len)
        train_cleaned_data.append(only_known_words)
        train_labels.append(d[0])
#        train_cleaned_data.append(doc2idx(dictionary, document=d[2], unknown_word_index=VOCAB_SIZE-1))
    test_cleaned_data = []
    test_labels = []
    test_num_words = []
    for d in test_data:
        only_known_words = list(filter((-1).__ne__, doc2idx(dictionary, document=d[1])))# should not include unknown words
        temp_len = len(only_known_words)
        if temp_len == 0:
            continue
        test_num_words.append(temp_len)
        test_cleaned_data.append(only_known_words)
        test_labels.append(d[0])

    print(len(test_cleaned_data), len(test_labels))
    return np.array(train_cleaned_data), np.array(train_labels), np.array(train_num_words), dictionary, np.array(test_cleaned_data), np.array(test_labels), np.array(test_num_words)


def vectorized_sample_data(D, y, num_words_array, num_samples, weight_vector='balanced', TESTING_FLAG=False): # maybe use pandas in future
    # do sampling without replacement. 
    if TESTING_FLAG == False:
        S_indices = np.random.choice(range(len(D)), num_samples, replace=False)
    else: # just same as range
        S_indices = np.arange(len(D))
    if USE_CUDA == False:
        Xs = Variable(torch.from_numpy(D[S_indices, :].astype(np.int, copy=False)))
        ys = Variable(torch.from_numpy(y[S_indices].astype(np.int, copy=False)))
#        Xs_num_words = Variable(torch.from_numpy(num_words_array[S_indices].astype(np.int, copy=False)))
        Xs_num_words = num_words_array[S_indices].astype(np.int, copy=False)
        weight_vector = Variable(torch.from_numpy(weight_vector[S_indices].astype(np.int, copy=False)))
    else :
        if TESTING_FLAG == True: # make data volatile (do not store the grads and all in GPU memory)
            Xs = Variable(torch.from_numpy(D[S_indices, :].astype(np.int, copy=False)).cuda(), volatile=True)
#            ys = Variable(torch.from_numpy(y[S_indices].astype(np.int, copy=False)).cuda(), volatile=True)
            ys = Variable(torch.from_numpy(y[S_indices].astype(np.float, copy=False)).type(torch.FloatTensor).cuda(), volatile=True)
            weight_vector = 'dummy'#Variable(torch.from_numpy(weight_vector[S_indices].astype(np.float, copy=False)).type(torch.FloatTensor).cuda())
        else:   
            Xs = Variable(torch.from_numpy(D[S_indices, :].astype(np.int, copy=False)).cuda())
            ys = Variable(torch.from_numpy(y[S_indices].astype(np.float, copy=False)).type(torch.FloatTensor).cuda())
            weight_vector = Variable(torch.from_numpy(weight_vector[S_indices].astype(np.float, copy=False)).type(torch.FloatTensor).cuda())
#            ys = Variable(torch.from_numpy(y[S_indices].astype(np.int, copy=False)).cuda())
#            Xs_num_words = Variable(torch.from_numpy(num_words_array[S_indices].astype(np.int, copy=False)).cuda())
        Xs_num_words = num_words_array[S_indices].astype(np.int, copy=False)
    return Xs, ys, Xs_num_words, weight_vector



#  operations are DATAPARALLEL 
class CoNN(torch.nn.Module): # version using inbuilt torch nn modules
    def __init__(self, name, hilbert_DIM, WORD2VEC, VOCAB_SIZE, TRUNC_LENGTH, dropout_p=DROPOUT): # initializing all the weights here
        super(CoNN, self).__init__() # initializing the nn.module
        self.WORD2VEC = WORD2VEC
        self.dropout_p  = dropout_p # randomly zeros some of the elements with probabaility = p: No dropout by default
        self.TRUNC_LENGTH = TRUNC_LENGTH
        if USE_CUDA == False: # shift to GPU
            self.dtype = torch.FloatTensor
        else: # shift to GPU
            print('shifting to cuda')
            self.dtype = torch.cuda.FloatTensor
        # wrapping with DataParallel
        self.fc1_theta = nn.Linear(hilbert_DIM, hilbert_DIM) # does Wx+b
        self.fc1_theta_bn = nn.BatchNorm1d(hilbert_DIM, affine=False) # do not save the normalization for testing: should be equal to the output dimension
#        self.fc2_theta = nn.Linear(hilbert_DIM, hilbert_DIM) # (input_size, output_size)
#        self.fc3_theta = nn.Linear(hilbert_DIM, hilbert_DIM) # (input_size, output_size)
        self.fc1_z     = nn.Linear(self.WORD2VEC+hilbert_DIM, hilbert_DIM)
#        self.fc1_z_bn  = nn.BatchNorm1d(2*hilbert_DIM, affine=False)
        self.fc1_z_bn  = nn.BatchNorm1d(self.TRUNC_LENGTH, affine=False)
#        self.fc2_z     = nn.Linear(hilbert_DIM, hilbert_DIM)
#        self.fc3_z     = nn.Linear(hilbert_DIM, hilbert_DIM)
#        self.fc_u      = nn.Linear(hilbert_DIM, num_CLASS, bias=False) # u (for loss)
        self.fc_u      = nn.Linear(hilbert_DIM, num_CLASS) # u (for loss)

        self.embed_dict = torch.nn.Embedding(VOCAB_SIZE, self.WORD2VEC) #initializing the embeddings
        self.dropout   = nn.Dropout(p=self.dropout_p)
        print('CoNN model ready')
        
    def forward(self, X, num_words, ITERATIONS, TEST_FLAG=0): # should return output embeddings, on which we can calculate the loss 
        # Vectorized implementation
        num_docs = X.data.shape[0]
        max_words = X.data.shape[1]
        mu_theta = Variable(torch.from_numpy(np.zeros((num_docs, hilbert_DIM))).type(self.dtype)) # should be faster ???
        mu_z = Variable(torch.from_numpy(np.zeros((num_docs, max_words, hilbert_DIM))).type(self.dtype))
        word_embedding = self.embed_dict(X).view(num_docs, max_words, self.WORD2VEC) #DxNxV
#        word_embedding = self.dropout(word_embedding)
        for t in range(ITERATIONS):
            mu_theta_clone = mu_theta.clone().view(num_docs,1, hilbert_DIM).repeat(1, max_words, 1).view(num_docs, max_words, hilbert_DIM)#DxH--> DxNxH
            concat_for_mu_z = torch.cat((word_embedding, mu_theta_clone), 2) # concat along last dim: should be DxNx(V+H)
            # mu_z --> DxNxH
#            mu_z = F.tanh(self.fc3_z(F.tanh(self.fc2_z(F.tanh(self.fc1_z_bn(self.fc1_z(concat_for_mu_z)))))))
#            mu_z = F.tanh(self.fc2_z(F.tanh(self.fc1_z_bn(self.fc1_z(concat_for_mu_z)))))
            mu_z = F.tanh(self.fc1_z_bn(self.fc1_z(concat_for_mu_z)))
            if TEST_FLAG==0:# training phase
                mu_z = F.dropout(mu_z, p=self.dropout_p, training=True)
            for d in range(num_docs):
                N = num_words[d]
                sum_mu_z_d = torch.sum(mu_z[d][:N], dim=0)
#                mu_theta[d] = F.tanh(self.fc3_theta(F.tanh(self.fc2_theta(F.tanh(self.fc1_theta_bn(self.fc1_theta(sum_mu_z_d)))))))
#                mu_theta[d] = F.tanh(self.fc2_theta(F.tanh(self.fc1_theta_bn(self.fc1_theta(sum_mu_z_d)))))
                mu_theta[d] = F.tanh(self.fc1_theta_bn(self.fc1_theta(sum_mu_z_d)))
                if TEST_FLAG==0:
                    mu_theta[d] = F.dropout(mu_theta[d], p=self.dropout_p, training=True)
        return self.fc_u(mu_theta) # returning output vector to calculate loss (take log loss of it!)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m*60
    return '%dm %ds' %(m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s # remaining time? 
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def numpy_fillna(data): # pads with zeros..
    data = np.array(data)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def truncate_data(D, w, length):
    if length >= D.shape[1]:
        print('Truncation length greater than the max number of words')
        return D, w
    else:
        w[w>length] = length
        return D[:, :length], w 

def cost_sensitive_weights(labels):
    df = pd.Series(labels)
#    print(df, df.value_counts())
    ratio = df.value_counts().tolist() #[0, 1]
    index = df.value_counts().index.values
#    print(ratio, index)
    imbalance_details = {}
    for i in range(len(ratio)):
        imbalance_details[int(index[i])] = float(ratio[0]+ratio[1])/float(ratio[i])
    print('imbalance details', imbalance_details)
    wt_vector = np.ones(len(labels))
#    print(imbalance_details[0], imbalance_details[1], labels[20])
    for i, l in enumerate(labels):
        wt_vector[i] = imbalance_details[int(l)] * wt_vector[i]
#    wt_vector = df.value_counts().tolist()
#    n1 = wt_vector / np.linalg.norm(wt_vector)
#    print(wt_vector)
    if IMBALANCE_HANDLING == True:
        return wt_vector # inverse weightage : PLAY with this in future
    else:
        return np.ones(len(labels))

def main(): 
    print("Document classification for Multi-domain sentimental dataset")
    basefile = 'sorted_data' # folder containing 342k data
    if PREP_DATA == True:# prepare data?>
        print('cleaning the train data')
        cleaned_data = prepare_data(basefile)
        print('saving the prepared data')
        prepared_data_list = cleaned_data#[train_cleaned_data, train_labels, train_num_words, dictionary, test_cleaned_data, test_labels, test_num_words]
        with open("preprocessed_multisent_data.txt", "wb") as f:
            pickle.dump(prepared_data_list, f)
    else:
        print('Loading the prepared data')
        with open("preprocessed_multisent_data.txt", "rb") as f:
            prepared_data_list = pickle.load(f)
        cleaned_data = prepared_data_list
    print(len(cleaned_data))
    print(len(cleaned_data[0]))
#    cleaned_data = cleaned_data[0]
#    M = nn.LogSoftmax()
#    criterion = nn.NLLLoss()
    #**********************
    # Just for doing K fold CV
    dataX = []
    dataY = []
    for d in cleaned_data:
        dataX.append(d[1])
        dataY.append(d[0])
        
    #**********************
    
#    weight_vector = torch.ones(1, num_CLASS)
#    if USE_CUDA == True:
#    	weight_vector = torch.ones(1, num_CLASS).cuda()

#    criterion = nn.CrossEntropyLoss(weight=weight_vector)# weights: tensor for weights for each class
    start = time.time()

    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=None) # Change random_state to 'int', act as a seed
    skf.get_n_splits(dataX, dataY)
    print(skf)
    k=0
    valid_loss = []
    valid_auc  = []
    # Run for these many batches (ref Alg 2)
    num_batches = NUM_BATCHES#400# 1000#2000 # (T)
    batch_size = BATCH_SIZE#1000 # (D_s)
    if batch_size%4 != 0:
        print("********POSSIBLE ERROR, as the data may not be properly divisble over all the GPUs ")    
    learning_rate = args.lr #0.05# 0.05
    accuracy_data = []
#    print(dataY)
    loss_plot = {}
#    accuracy_plot = {}
    auc_plot = {}
    if DO_CV == True: 
        for train_index, valid_index in skf.split(dataX, dataY):
            loss_plot[k] = []
#            accuracy_plot[k] = []
            auc_plot[k] = []
            #******************** get the train and test dataset
            train_cleaned_data, train_labels, train_num_words, dictionary, valid_cleaned_data, valid_labels, valid_num_words = get_CV_data(cleaned_data, train_index, valid_index)
            print('value counts for valid set')
            bc = np.bincount(train_labels)
            ii = np.nonzero(bc)[0]
            print(np.vstack((ii, bc[ii])).T)
#****************************************
            print('Getting different weights for cost sensitive learning')
            weight_vector = cost_sensitive_weights(train_labels)
#            weight_vector = Variable(torch.from_numpy(weight_vector))
#            if USE_CUDA == True:
#                weight_vector = Variable(torch.from_numpy(weight_vector.astype(np.float, copy=False)).type(torch.FloatTensor).cuda())
#            else:
#                weight_vector = Variable(torch.from_numpy(weight_vector))
 

#            criterion = nn.CrossEntropyLoss(weight=weight_vector)# weights: tensor for weights for each class
#****************************************

            train_cleaned_data = numpy_fillna(train_cleaned_data)
#            TRUNC_LENGTH = int(2*np.mean(train_num_words))
            TRUNC_LENGTH = int(np.mean(train_num_words)) + 50
#            TRUNC_LENGTH = int(np.median(train_num_words))# + 50
            print('Trucation length', TRUNC_LENGTH)
            train_cleaned_data, train_num_words = truncate_data(train_cleaned_data, train_num_words, TRUNC_LENGTH) # max_words in a Document
            VOCAB_SIZE = int(len(dictionary)) + 1
            #***************************************************

            # Re-initialise the model
            # initialise the parameters of the model
            if MODEL_NAME is 'CoNN':
                model = CoNN('nn_deep', hilbert_DIM, WORD2VEC, VOCAB_SIZE, TRUNC_LENGTH)
            else: 
                print("model not present")

            if USE_CUDA == True:
                model = model.cuda() #:TODO: Does this work!! maybe put NN modules on GPU...
                print('WRAPPING aournd DataParallel.. ')
                model = nn.DataParallel(model)
            if MODEL_NAME is 'CoNN' and args.optimizer=='adam':
    #            optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)#, amsgrad=False)
    #            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.95, centered=False)
    #            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,  dampening=0, weight_decay=0, nesterov=False)

            print('Adding Scheduler')
    #        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    #        scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
            scheduler = MultiStepLR(optimizer, milestones=[1000, 1200, 1500, 1800, 2400, 3000, 3500], gamma=0.5)
    #        scheduler = ExponentialLR(optimizer, gamma=0.1)

            print('K_fold = ', k)
            for t in range(1, num_batches+1): # change to number of epochs\
                scheduler.step()
                if t%100 == 0: # print learning rate after every 100 epochs
                    for param_group in optimizer.param_groups:
                        print('lr ', param_group['lr'])
                # sample docs from train_cleaned_data
                Xs, ys, Xs_num_words, batch_weight_vector = vectorized_sample_data(train_cleaned_data, train_labels, train_num_words, batch_size, weight_vector) # returns  
                optimizer.zero_grad()   # zero the gradient buffers
                output = model.forward(Xs, Xs_num_words, ITERATIONS)# DROPOUT of p 
    #            loss = criterion(M(output), ys)
#                loss = criterion(output, ys)
                loss = F.binary_cross_entropy_with_logits(output.view(len(ys)), ys, weight=batch_weight_vector) # NOTE: use weights for cost sensitive learning
                # calculating the accuracy
                ys_actual   = ys.data.cpu().numpy()
                scores_pred = output.data.cpu().numpy()
#                ys_pred     = np.argmax(scores_pred, axis=1)
#                accuracy    = accuracy_score(ys_actual, ys_pred)*100
                auc         = get_auc(ys_actual, scores_pred) 
                print('time since start ', time_since(start, float(t)/float(num_batches)),'(%d %d%%)'%(t, float(t)/float(num_batches)*100 ), 'loss', loss.data.cpu().numpy(), ' auc ', auc)
                loss_plot[k].append(loss.data.cpu().numpy())
#                accuracy_plot[k].append(accuracy)
                auc_plot[k].append(auc)
                loss.backward()
                optimizer.step()
    #        scheduler.step(loss.data[0])
                if EVALUATE_TEST == True and t%EVALUATE_EVERY==0: # evaluate the model after every 100 iterations
                    if t == EVALUATE_EVERY:
                        valid_cleaned_data = numpy_fillna(valid_cleaned_data)
                        valid_cleaned_data, valid_num_words = truncate_data(valid_cleaned_data, valid_num_words, TRUNC_LENGTH) # max_words in a Document
                        Xt, yt, Xt_num_words, _ = vectorized_sample_data(valid_cleaned_data, valid_labels, valid_num_words, 'dummy_len(valid_index)', TESTING_FLAG=True, weight_vector='dummy') # Hoping that it runs for all the validation set at once!!
                    # NOTE: maintaining the same batch size for validation too
                    output = []
                    num_valid_batches = int(np.ceil(len(yt)/batch_size))
                    for b in range(num_valid_batches):
                        if b == num_valid_batches-1:
                            batch_output = model.forward(Xt[batch_size*b:, :], Xt_num_words[batch_size*b:], ITERATIONS, TEST_FLAG=1)
                            output.append(batch_output.data.cpu().numpy()) # NOTE: default dropout is '0' for valid and test case! (as we need to take the ensemble of DNNs)
                        else:
                            batch_output = model.forward(Xt[batch_size*b:batch_size*(b+1), :], Xt_num_words[batch_size*b:batch_size*(b+1)], ITERATIONS, TEST_FLAG=1)
                            output.append(batch_output.data.cpu().numpy()) # NOTE: default dropout is '0' for valid and test case! (as we need to take the ensemble of DNNs)
                    output = list(itertools.chain.from_iterable(output))
                    total_valid_output = Variable(torch.from_numpy(np.array(output)).type(torch.cuda.FloatTensor))
                    loss = F.binary_cross_entropy_with_logits(total_valid_output.view(len(yt)), yt, weight=None)
                    # calculating the accuracy
                    yt_actual   = yt.data.cpu().numpy()
                    scores_pred = output#.data.cpu().numpy()
                    auc         = get_auc(yt_actual, scores_pred) 
                    print('Valid loss for k =', k, ' Iteration num ', t, '\n loss ', loss.data.cpu().numpy()[0], '\n auc ', auc)
            loss_plot[k] = np.vstack(loss_plot[k])
            loss_plot[k] = loss_plot[k][:, 0]
            print('checking loss on validation set')
            valid_cleaned_data = numpy_fillna(valid_cleaned_data)
            valid_cleaned_data, valid_num_words = truncate_data(valid_cleaned_data, valid_num_words, TRUNC_LENGTH) # max_words in a Document
            Xs, ys, Xs_num_words, _ = vectorized_sample_data(valid_cleaned_data, valid_labels, valid_num_words, 'dummy_len(valid_index)', TESTING_FLAG=True, weight_vector='dummy') # Hoping that it runs for all the validation set at once!!
            # NOTE: maintaining the same batch size for validation too
            output = []
            num_valid_batches = int(np.ceil(len(ys)/batch_size))
    #        batch_loss = [] 
            for b in range(num_valid_batches):
    #            print('valid data, batch_num ', b)
                if b == num_valid_batches-1:
                    batch_output = model.forward(Xs[batch_size*b:, :], Xs_num_words[batch_size*b:], ITERATIONS, TEST_FLAG=1)
    #                batch_loss.append(criterion(batch_output, ys[batch_size*b:]).data.cpu().numpy())
                    output.append(batch_output.data.cpu().numpy()) # NOTE: default dropout is '0' for valid and test case! (as we need to take the ensemble of DNNs)
                else:
                    batch_output = model.forward(Xs[batch_size*b:batch_size*(b+1), :], Xs_num_words[batch_size*b:batch_size*(b+1)], ITERATIONS, TEST_FLAG=1)
    #                batch_loss.append(criterion(batch_output, ys[batch_size*b:batch_size*(b+1)]).data.cpu().numpy())
                    output.append(batch_output.data.cpu().numpy()) # NOTE: default dropout is '0' for valid and test case! (as we need to take the ensemble of DNNs)
            output = list(itertools.chain.from_iterable(output))
    #        loss   = np.array(list(itertools.chain.from_iterable(batch_loss))).sum()
            total_valid_output = Variable(torch.from_numpy(np.array(output)).type(torch.cuda.FloatTensor))
#            loss = criterion(total_valid_output, ys)
            loss = F.binary_cross_entropy_with_logits(total_valid_output.view(len(ys)), ys, weight=None)
            # calculating the accuracy
            ys_actual   = ys.data.cpu().numpy()
            scores_pred = output#.data.cpu().numpy()
#            ys_pred     = np.argmax(scores_pred, axis=1)
#            accuracy    = accuracy_score(ys_actual, ys_pred)*100  
#            accuracy_data.append(accuracy)
            auc         = get_auc(ys_actual, scores_pred) 
            print('Valid loss for k =', k, ' is ', loss, '\n', ' auc ', auc)
            valid_loss.append(loss.data.cpu().numpy()) # loss.data
            valid_auc.append(auc)
    #        valid_loss.append(loss) # loss.data
            k+=1

        valid_loss = np.vstack(valid_loss)
        valid_loss = valid_loss[:, 0]
        print('K-fold valid loss ', valid_loss)
        print('K-fold auc ', valid_auc)
        print('avg auc ', np.array(valid_auc).mean(), ' std dev', np.array(valid_auc).std())
#        for k in range(K_FOLD):
#            print('fold ', k, ' ' )
 
        with open('cv_results_342k_multisent', 'wb') as fp:
            pickle.dump([valid_loss, valid_auc, loss_plot, auc_plot], fp)
    #    with open('accuracy_data', 'rb') as fp:
    #        accuracy_data = pickle.load(fp)

    print('saving the model')
    torch.save(model, 'CoNN_model_342k_multisent.pt')
    return

if __name__=="__main__":
    main()
