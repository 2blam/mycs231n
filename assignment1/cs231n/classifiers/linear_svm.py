import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) #(10, )
    correct_class_score = scores[y[i]]
    counter = 0
    for j in xrange(num_classes):  
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        counter = counter + 1
        dW[:, j] = dW[:, j] + X[i, :]

    dW[:, y[i]] = dW[:, y[i]] -  X[i, :] * counter #correct class
    


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  score = X.dot(W) #500, 10
  cscore =  score[xrange(num_train), y] #500, 0
  margin = np.maximum(0, score.T - cscore + 1) #correct labelled element, now with value 1
  loss = (np.sum(margin) - num_train) / num_train  # minus num_train due to the added extra value 1
  loss += 0.5 * reg * np.sum(W * W)  

  # cscore = score[xrange(score.shape[0]), y]   #get the correct score
  # cscore = cscore.reshape((-1, 1))               #convert to column vector (500, 1)
  # cscore = np.repeat(cscore, W.shape[1], axis=1) #replicate correct score form (500, 10) matrix  
  # diff = score - cscore + 1 
  # diff[diff < 0] = 0                             #stimulate max function
  # diff[np.arange(diff.shape[0]), y] = 0          #put 0 to correct labelled elements
  # loss = np.sum(diff) 
  # loss /= X.shape[0]
  # loss += 0.5 * reg * np.sum(W * W)  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  temp = np.zeros((margin.shape))
  temp[margin>0] = 1
  temp[y, xrange(num_train)] -= np.sum(margin>0, axis=0)  
  dW = X.T.dot(temp.T) / num_train  
  dW += reg * W 



  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
