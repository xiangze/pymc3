from __future__ import division
from theano.tensor import constant, flatten, zeros_like, ones_like, stack, concatenate, sum, prod, lt, gt, le, ge, eq, neq, switch, clip, where, and_, or_, abs_

from theano.tensor import exp, log, cos, sin, tan, cosh, sinh, tanh, sqr, sqrt, erf, erfinv, dot
from theano.tensor import maximum, minimum, sgn, ceil, floor

from theano.tensor.nlinalg import det, matrix_inverse, extract_diag, matrix_dot, trace
from theano.tensor.nnet import sigmoid
import theano
from theano import ifelse
import theano.tensor as tt
import sys

def invlogit(x):
    x_max = -tt.log(sys.float_info.epsilon)
    x = ifelse(tt.gt(x, x_max), 1.0, ifelse(tt.lt(x, 1-x_max), 0.0, x))
    return 1/(1 + tt.exp(-x))
    
def logit(p):
    return tt.log(p/(1-p))