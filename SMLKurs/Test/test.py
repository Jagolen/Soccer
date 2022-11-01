import numpy as np
import pandas as pd
#Tenta senaste


# 1 a


x = np.array([6, 10, 8, 11, 7])
y = np.array([1, -1, 1, -1, 1])
beta0=9
beta1=-1
adj_x= x*beta1 + beta0
l=np.exp(adj_x*y)/(1+np.exp(adj_x*y))
print(l)
likelihood=np.prod(l)
loglikelihood=np.sum(np.log(l))
print(loglikelihood)

#1 b

upper_r =1/(1+np.exp(10*beta1+beta0))
lower_r =1/(1+np.exp(8*beta1+beta0))

print(f'{lower_r} < r < {upper_r}')

#1 c

r = 0.6
xc = np.array([8, 9, 11, 7, 12])
y = np.array([1, -1, 1, 1, -1])
ytest = [0, 0, 0, 0, 0]
for i in range(5):
    ytest[i] = (np.exp(beta0 + beta1 * x[i]))/(1+np.exp(beta0 + beta1 * x[i]))
    if ytest[i] > r:
        ytest[i] = 1
    else:
        ytest[i] = -1
print(f'ytest = {ytest}')
misser = np.mean(y != ytest)

p = 0
fp = 0
tp = 0
n = 0
for i in range(5):
    if y[i] == 1:
        p += 1
        if ytest[i] == 1:
            tp += 1
    else:
        n += 1
        if ytest[i] == 1:
            fp += 1
tp = tp/p
fp = fp/n
print(f'True positive: {tp}, False positive: {fp}, Missclassification rate: {misser}')

