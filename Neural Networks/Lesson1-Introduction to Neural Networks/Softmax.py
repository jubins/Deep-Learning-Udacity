'''
Python 3
@jubin-soni
The softmax function is the equivalent of the sigmoid activation function, but when the problem has 3 or more classes.
In softmax we use exponential function as it always returns positive values.

Softmax Formula:
P(class i) = e^Zi / (e^Z1 + e^Z2 + ... + e^Zn)

'''
import numpy as np
# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

def softmax(L):
    sum_z = sum(np.exp(L))
    softmax_score = [np.exp(Zi)/sum_z for Zi in L]
    return softmax_score


def main():
    #Test case#1 (Udacity)
    L = [5, 6, 7]
    #Udacity output: [0.090030573170380462, 0.24472847105479764, 0.6652409557748219]
    print(softmax(L))

    #Test case#2
    L = [1,2,3,4,5,6,7,8,9]
    print(softmax(L))


if __name__ == '__main__':
    main()
