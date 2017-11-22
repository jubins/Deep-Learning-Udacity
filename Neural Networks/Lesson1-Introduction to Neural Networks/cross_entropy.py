'''
Python 3
@jubin-soni

Cross Entropy tells that if there are bunch of events and if those events have probabilities, how likely it is that
those events will happen based on the probabilities. If it is very likely then we have a small cross-entropy, and if it
is unlikely then we have a large cross-entropy.

Cross Entropy Formula:
CE = -sigma(i to m) Yi*ln(Pi) + (1-Yi)*ln(1-Pi) ; where Yi*ln(Pi) is probability of likelihood with Yi being 0 in (1-Yi)
                                             and (1-Yi)*ln(1-Pi) is probability of unlikelihood with Yi being 0 in Yi
'''
import numpy as np
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    m = list(zip(Y, P))
    CE = 0
    for i in m:
        Yi, Pi = i[0], i[1]
        CE += -((Yi*np.log(Pi))+((1-Yi)*np.log(1-Pi)))
    return CE

def main():
    #Test case#1 (Udacity)
    Y = [1, 0, 1, 1]
    P = [0.4, 0.6, 0.1, 0.5]
    #Expected (Udacity) output: 4.8283137373
    print(cross_entropy(Y, P))

    #Test case#2
    Y = [1, 1, 0, 0]
    P = [0.4, 0.6, 0.1, 0.5]
    print(cross_entropy(Y, P))

if __name__ == '__main__':
    main()
