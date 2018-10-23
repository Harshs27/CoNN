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
from model import CoNN

import argparse

parser = argparse.ArgumentParser(description='Running Cooperative Neural Network (CoNN-sLDA)\
                                                                 for document classification')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--MODEL_NAME', type=str, default='CoNN', 
                    help='Using the CoNN-sLDA model')
parser.add_argument('--USE_CUDA', type=str2bool, nargs='?', default=True, 
                    help='use GPUs if available, uses all available gpu by default:\
                           If you want to run on specific GPUs, use CUDA_VISIBLE_DEVICES')
parser.add_argument('--PREP_DATA', type=str2bool, nargs='?', default=False, 
                    help='preprocess the data, the script saves the processed data for later reuse')
parser.add_argument('--DO_CV', type=str2bool, nargs='?', default=True, 
                    help='do cross validation')
parser.add_argument('--K_FOLD', type=int, default=5,
                    help='K fold validation')
parser.add_argument('--NUM_BATCHES', type=int, default=3000,#1200, #3000,
                    help='Number of batches for training')
parser.add_argument('--BATCH_SIZE', type=int, default=100,
                    help='Batch size')
parser.add_argument('--DROPOUT', type=float, default=0,
                    help='dropout for regularization')
parser.add_argument('--IMBALANCE_HANDLING', type=str2bool, nargs='?', default=False, 
                    help='apply cost-sensitive learning to the loss function')
parser.add_argument('--EVALUATE_TEST', type=str2bool, nargs='?', default=True, 
                    help='Report the results on validation dataset at EVALUATE_EVERY iterations')
parser.add_argument('--EVALUATE_EVERY', type=int, default=500,
                    help='evaluate at intervals of EVALUATE_EVERY epochs')
parser.add_argument('--hilbert_DIM', type=int, default=10,
                    help='The size of the hilbert space embedding vector')
parser.add_argument('--WORD2VEC', type=int, default=10,
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


def doc2idx(dictionary, document, unknown_word_index=-1):
        if isinstance(document, string_types):
            raise TypeError("doc2idx expects an array of unicode tokens on input, not a single string")
        document = [word if isinstance(word, str) else unicode(word, 'utf-8') for word in document]
        return [dictionary.token2id.get(word, unknown_word_index) for word in document]


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

    return np.array(train_cleaned_data), np.array(train_labels), np.array(train_num_words),\
             dictionary, np.array(test_cleaned_data), np.array(test_labels), np.array(test_num_words)


def vectorized_sample_data(D, y, num_words_array, num_samples, weight_vector='balanced', TESTING_FLAG=False): 
    # do sampling without replacement. 
    if TESTING_FLAG == False:
        S_indices = np.random.choice(range(len(D)), num_samples, replace=False)
    else: # just same as range
        S_indices = np.arange(len(D))
    if args.USE_CUDA == False:
        Xs = Variable(torch.from_numpy(D[S_indices, :].astype(np.int, copy=False)))
        ys = Variable(torch.from_numpy(y[S_indices].astype(np.int, copy=False)))
        Xs_num_words = num_words_array[S_indices].astype(np.int, copy=False)
        weight_vector = Variable(torch.from_numpy(weight_vector[S_indices].astype(np.int, copy=False)))
    else :
        if TESTING_FLAG == True: # make data volatile (do not store the grads and all in GPU memory)
            Xs = Variable(torch.from_numpy(D[S_indices, :].astype(np.int, copy=False)).cuda(), volatile=True)
            ys = Variable(torch.from_numpy(y[S_indices].astype(np.float, copy=False))\
                                                              .type(torch.FloatTensor).cuda(), volatile=True)
            weight_vector = 'dummy'#Variable(torch.from_numpy(weight_vector[S_indices].astype(np.float, copy=False)).type(torch.FloatTensor).cuda())
        else:   
            Xs = Variable(torch.from_numpy(D[S_indices, :].astype(np.int, copy=False)).cuda())
            ys = Variable(torch.from_numpy(y[S_indices].astype(np.float, copy=False))\
                                                                             .type(torch.FloatTensor).cuda())
            weight_vector = Variable(torch.from_numpy(weight_vector[S_indices].astype(np.float, copy=False))\
                                                                             .type(torch.FloatTensor).cuda())
        Xs_num_words = num_words_array[S_indices].astype(np.int, copy=False)
    return Xs, ys, Xs_num_words, weight_vector


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
    ratio = df.value_counts().tolist() #[0, 1]
    index = df.value_counts().index.values
    imbalance_details = {}
    for i in range(len(ratio)):
        imbalance_details[int(index[i])] = float(ratio[0]+ratio[1])/float(ratio[i])
    print('imbalance details', imbalance_details)
    wt_vector = np.ones(len(labels))
    for i, l in enumerate(labels):
        wt_vector[i] = imbalance_details[int(l)] * wt_vector[i]
#    wt_vector = df.value_counts().tolist()
#    n1 = wt_vector / np.linalg.norm(wt_vector)
    if args.IMBALANCE_HANDLING == True:
        return wt_vector # inverse weightage : PLAY with this in future
    else:
        return np.ones(len(labels))

def main(): 
    print("Document classification for Multi-domain sentimental dataset")
    basefile = 'sorted_data' # folder containing 342k data
    if args.PREP_DATA == True:# prepare data
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
    #**********************
    # Just for doing K fold CV
    dataX = []
    dataY = []
    for d in cleaned_data:
        dataX.append(d[1])
        dataY.append(d[0])
        
    #**********************
    start = time.time()
    skf = StratifiedKFold(n_splits=args.K_FOLD, shuffle=True, random_state=None) # Change random_state to 'int', act as a seed
    skf.get_n_splits(dataX, dataY)
    print(skf)
    k=0
    valid_loss = []
    valid_auc  = []
    # Run for these many batches (ref Alg 2)
    num_batches = args.NUM_BATCHES
    batch_size = args.BATCH_SIZE
    if batch_size%4 != 0:
        print("********POSSIBLE ERROR, as the data may not be properly divisble over all the GPUs ")    
    learning_rate = args.lr 
    accuracy_data = []
    loss_plot = {}
    auc_plot = {}
    if args.DO_CV == True: 
        for train_index, valid_index in skf.split(dataX, dataY):
            loss_plot[k] = []
            auc_plot[k] = []
            #******************** get the train and test dataset
            train_cleaned_data, train_labels, train_num_words, dictionary, valid_cleaned_data, valid_labels, valid_num_words\
                                                                        = get_CV_data(cleaned_data, train_index, valid_index)
            print('value counts for valid set')
            bc = np.bincount(train_labels)
            ii = np.nonzero(bc)[0]
            print(np.vstack((ii, bc[ii])).T)

            print('Getting different weights for cost sensitive learning')
            weight_vector = cost_sensitive_weights(train_labels)

            train_cleaned_data = numpy_fillna(train_cleaned_data)
#            TRUNC_LENGTH = int(2*np.mean(train_num_words))
            TRUNC_LENGTH = int(np.mean(train_num_words)) + 50
#            TRUNC_LENGTH = int(np.median(train_num_words))# + 50
            print('Trucation length, done for vectorization of code', TRUNC_LENGTH)
            train_cleaned_data, train_num_words = truncate_data(train_cleaned_data, train_num_words, TRUNC_LENGTH) # max_words in a Document
            VOCAB_SIZE = int(len(dictionary)) + 1
            #***************************************************
            # Re-initialise the model
            # initialise the parameters of the model
            if args.MODEL_NAME is 'CoNN':
                model = CoNN(args.MODEL_NAME, args.hilbert_DIM, args.WORD2VEC, VOCAB_SIZE, TRUNC_LENGTH, 
                                        dropout_p=args.DROPOUT, USE_CUDA=args.USE_CUDA, num_CLASS=args.num_CLASS)
            else: 
                print("model not present")

            if args.USE_CUDA == True:
                model = model.cuda() #:TODO: Does this work!! maybe put NN modules on GPU...
                print('WRAPPING around DataParallel.. ')
                model = nn.DataParallel(model)
            if args.MODEL_NAME is 'CoNN' and args.optimizer=='adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), \
                                                                              eps=1e-08, weight_decay=0)

            print('Adding Scheduler')
            scheduler = MultiStepLR(optimizer, milestones=[1000, 1200, 1500, 1800, 2400, 3000, 3500], gamma=0.5)

            print('K_fold = ', k)
            for t in range(1, num_batches+1): # change to number of epochs\
                scheduler.step()
                if t%100 == 0: # print learning rate after every 100 epochs
                    for param_group in optimizer.param_groups:
                        print('lr ', param_group['lr'])



                #******************************************************************************
                #**************************TRAINING********************************************
                #******************************************************************************
                # sample docs from train_cleaned_data
                Xs, ys, Xs_num_words, batch_weight_vector = vectorized_sample_data(train_cleaned_data, train_labels,\
                                                                            train_num_words, batch_size, weight_vector) 
                optimizer.zero_grad()   # zero the gradient buffers
                output = model.forward(Xs, Xs_num_words, args.ITERATIONS)# DROPOUT of p 
                loss = F.binary_cross_entropy_with_logits(output.view(len(ys)), ys, weight=batch_weight_vector) 
                # NOTE: use weights for cost sensitive learning
                # calculating the accuracy
                ys_actual   = ys.data.cpu().numpy()
                scores_pred = output.data.cpu().numpy()
                auc         = get_auc(ys_actual, scores_pred) 
                print('time since start ', time_since(start, float(t)/float(num_batches)),'(%d %d%%)'\
                                  %(t, float(t)/float(num_batches)*100 ), 'loss', loss.data.cpu().numpy(), ' auc ', auc)
                loss_plot[k].append(loss.data.cpu().numpy())
                auc_plot[k].append(auc)
                loss.backward()
                optimizer.step()
                #******************************************************************************
                #****************************END***********************************************



                #******************************************************************************
                #****************************INTERMEDIATE EVALUATION***************************
                #******************************************************************************
                if args.EVALUATE_TEST == True and t%args.EVALUATE_EVERY==0: # evaluate the model after every 100 iterations
                    if t == args.EVALUATE_EVERY:
                        valid_cleaned_data = numpy_fillna(valid_cleaned_data)
                        valid_cleaned_data, valid_num_words = truncate_data(valid_cleaned_data, valid_num_words,\
                                                                                                     TRUNC_LENGTH) 
                        Xt, yt, Xt_num_words, _ = vectorized_sample_data(valid_cleaned_data, \
                                                  valid_labels, valid_num_words, 'dummy_len(valid_index)',\
                                                                  TESTING_FLAG=True, weight_vector='dummy') 
                        # Hoping that it runs for all the validation set at once!!
                    # NOTE: maintaining the same batch size for validation too
                    output = []
                    num_valid_batches = int(np.ceil(len(yt)/batch_size))
                    for b in range(num_valid_batches):
                        if b == num_valid_batches-1:
                            batch_output = model.forward(Xt[batch_size*b:, :],\
                                                  Xt_num_words[batch_size*b:], args.ITERATIONS, TEST_FLAG=1)
                            output.append(batch_output.data.cpu().numpy()) 
                            # NOTE: default dropout is '0' for valid and test case! (as we need to take the ensemble of DNNs)
                        else:
                            batch_output = model.forward(Xt[batch_size*b:batch_size*(b+1), :],\
                                          Xt_num_words[batch_size*b:batch_size*(b+1)], args.ITERATIONS, TEST_FLAG=1)
                            output.append(batch_output.data.cpu().numpy()) 
                            # NOTE: default dropout is '0' for valid and test case! (as we need to take the ensemble of DNNs)
                    output = list(itertools.chain.from_iterable(output))
                    total_valid_output = Variable(torch.from_numpy(np.array(output)).type(torch.cuda.FloatTensor))
                    loss = F.binary_cross_entropy_with_logits(total_valid_output.view(len(yt)), yt, weight=None)
                    # calculating the accuracy
                    yt_actual   = yt.data.cpu().numpy()
                    scores_pred = output#.data.cpu().numpy()
                    auc         = get_auc(yt_actual, scores_pred) 
                    print('Valid loss for k =', k, ' Iteration num ', t, '\n loss ', \
                                                   loss.data.cpu().numpy()[0], '\n auc ', auc)
                #******************************************************************************
                #*****************************END**********************************************


            loss_plot[k] = np.vstack(loss_plot[k])
            loss_plot[k] = loss_plot[k][:, 0]

            # NOTE: maintaining the same batch size for validation too
            #******************************************************************************
            #**************************VALIDATION******************************************
            #******************************************************************************
            print('checking loss on validation set')
            valid_cleaned_data = numpy_fillna(valid_cleaned_data)
            valid_cleaned_data, valid_num_words = truncate_data(valid_cleaned_data,\
                                                              valid_num_words, TRUNC_LENGTH) # max_words in a Document
            Xs, ys, Xs_num_words, _ = vectorized_sample_data(valid_cleaned_data,\
                                       valid_labels, valid_num_words, 'dummy_len(valid_index)',\
                                        TESTING_FLAG=True, weight_vector='dummy') 
            # Hoping that it runs for all the validation set at once!!
            output = []
            num_valid_batches = int(np.ceil(len(ys)/batch_size))
            for b in range(num_valid_batches):
                if b == num_valid_batches-1:
                    batch_output = model.forward(Xs[batch_size*b:, :],\
                                       Xs_num_words[batch_size*b:], args.ITERATIONS, TEST_FLAG=1)
                    output.append(batch_output.data.cpu().numpy()) 
                    # NOTE: default dropout is '0' for valid and test case! (as we need to take the ensemble of DNNs)
                else:
                    batch_output = model.forward(Xs[batch_size*b:batch_size*(b+1), :],\
                                  Xs_num_words[batch_size*b:batch_size*(b+1)], args.ITERATIONS, TEST_FLAG=1)
                    output.append(batch_output.data.cpu().numpy()) 
                    # NOTE: default dropout is '0' for valid and test case! (as we need to take the ensemble of DNNs)
            output = list(itertools.chain.from_iterable(output))
            total_valid_output = Variable(torch.from_numpy(np.array(output)).type(torch.cuda.FloatTensor))
            loss = F.binary_cross_entropy_with_logits(total_valid_output.view(len(ys)), ys, weight=None)
            ys_actual   = ys.data.cpu().numpy()
            scores_pred = output#.data.cpu().numpy()
            auc         = get_auc(ys_actual, scores_pred) 
            print('Valid loss for k =', k, ' is ', loss, '\n', ' auc ', auc)
            valid_loss.append(loss.data.cpu().numpy()) # loss.data
            valid_auc.append(auc)
            #******************************************************************************
            #**************************VALIDATION end**************************************
            #******************************************************************************
            k+=1

        valid_loss = np.vstack(valid_loss)
        valid_loss = valid_loss[:, 0]
        print('K-fold valid loss ', valid_loss)
        print('K-fold auc ', valid_auc)
        print('avg auc ', np.array(valid_auc).mean(), ' std dev', np.array(valid_auc).std())
        with open('cv_results_multisent', 'wb') as fp:
            pickle.dump([valid_loss, valid_auc, loss_plot, auc_plot], fp)

    print('saving the model')
    torch.save(model, 'CoNN_model_multisent.pt')
    return

if __name__=="__main__":
    main()
