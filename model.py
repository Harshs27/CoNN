import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#  operations are DATAPARALLEL 
class CoNN(torch.nn.Module): # version using inbuilt torch nn modules
    def __init__(self, name, hilbert_DIM, WORD2VEC, VOCAB_SIZE, TRUNC_LENGTH, dropout_p=0, USE_CUDA=True, num_CLASS=1): # initializing all the weights here
        super(CoNN, self).__init__() # initializing the nn.module
        self.WORD2VEC = WORD2VEC
        self.hilbert_DIM = hilbert_DIM
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
        mu_theta = Variable(torch.from_numpy(np.zeros((num_docs, self.hilbert_DIM))).type(self.dtype)) # should be faster ???
        mu_z = Variable(torch.from_numpy(np.zeros((num_docs, max_words, self.hilbert_DIM))).type(self.dtype))
        word_embedding = self.embed_dict(X).view(num_docs, max_words, self.WORD2VEC) #DxNxV
#        word_embedding = self.dropout(word_embedding)
        for t in range(ITERATIONS):
            mu_theta_clone = mu_theta.clone().view(num_docs,1, self.hilbert_DIM).repeat(1, max_words, 1).view(num_docs, max_words, self.hilbert_DIM)#DxH--> DxNxH
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
