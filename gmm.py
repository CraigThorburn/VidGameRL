#import pytorch
import numpy as np
from scipy.stats import multivariate_normal as mvn

N = 20

cat11_mean = .4
cat11_var = .3
cat12_mean = .5
cat12_var = .2
cat13 = 0.8
cat14 = 0.4

cat21_mean = .9
cat21_var = .3
cat22_mean = .1
cat22_var = .2
cat23 = 0.2
cat24 = 0.6


cat1_bin1 = np.random.rand(N) < cat13
cat1_bin1 = cat1_bin1.astype(int)
cat1_bin1 = np.array([cat1_bin1, 1-cat1_bin1])

cat1_bin2 = np.random.rand(N) < cat14
cat1_bin2 = cat1_bin2.astype(int)
cat1_bin2 = np.array([cat1_bin2, 1-cat1_bin2])


cat2_bin1 = np.random.rand(N) < cat23
cat2_bin1 = cat2_bin1.astype(int)
cat2_bin1 = np.array([cat2_bin1, 1-cat2_bin1])

cat2_bin2 = np.random.rand(N) < cat24
cat2_bin2 = cat2_bin2.astype(int)
cat2_bin2 = np.array([cat2_bin2, 1-cat2_bin2])

cat11 = np.random.normal(cat11_mean, cat11_var, N)
cat12 = np.random.normal(cat12_mean, cat12_var, N)
cat21 = np.random.normal(cat21_mean, cat21_var, N)
cat22 = np.random.normal(cat22_mean, cat22_var, N)

cats = np.array([np.concatenate((cat11,cat21)), np.concatenate((cat12,cat22))]).transpose()
cats_bin1 = np.concatenate((cat1_bin1, cat2_bin1), 1).transpose()
cats_bin2 = np.concatenate((cat1_bin2, cat2_bin2), 1).transpose()

#Initialization
num_cats = 2
gaussian_dims = 2
pi = np.ones(num_cats)/num_cats
mu = np.random.rand(num_cats,gaussian_dims)
sigma_random = np.random.rand(num_cats, gaussian_dims,gaussian_dims )
sigma = np.array([np.dot(sigma_random[i], sigma_random[i].transpose()) for i in range(num_cats)])
phi1_random = np.random.rand(num_cats)/2
phi2_random = np.random.rand(num_cats)/2

phi1 = np.array([phi1_random, 1-phi1_random]).transpose()
phi2= np.array([phi2_random, 1-phi2_random]).transpose()

epochs = 10
for x in range(epochs):
    #Expectation
    gs = [mvn(mean = mu[i], cov = sigma[i]) for i in range(num_cats)]
    g_outs = np.array([mvn.pdf(cats) for mvn in gs])
    fs1 = np.array([(phi1[i]*cats_bin1).sum(1) for i in range(num_cats)])
    fs2 = np.array([(phi2[i] * cats_bin2).sum(1) for i in range(num_cats)])
    zs_num = ((g_outs * fs1 * fs2).transpose()*pi)
    zs_den = np.sum(((g_outs * fs1 * fs2).transpose()*pi),1)

    zs = np.array([zs_num[i]/zs_den[i] for i in range(N*2)])

    #Maximization
    Nk = np.sum(zs,0)

    mu = np.array([np.sum(zs[:,i]*cats.transpose(),1)/Nk[i] for i in range(num_cats)])
    sigma = np.array([np.sum(zs[:,i]*np.dot((cats-mu[i]),(cats-mu[i]).transpose()),1)/Nk[i] for i in range(num_cats)])

    phi1 = np.array([np.sum(zs[:,i]*cats_bin1.transpose(),1)/Nk[i] for i in range(num_cats)])
    phi2 = np.array([np.sum(zs[:,i]*cats_bin2.transpose(),1)/Nk[i] for i in range(num_cats)])

    pi = Nk/(N*2)

    #Convergence Check
    gs = [mvn(mean=mu[i], cov=sigma[i]) for i in range(num_cats)]
    g_outs = np.array([mvn.pdf(cats) for mvn in gs])
    fs1 = np.array([(phi1[i] * cats_bin1).sum(1) for i in range(num_cats)])
    fs2 = np.array([(phi2[i] * cats_bin2).sum(1) for i in range(num_cats)])
    ll = sum(sum(zs*(np.log(pi) + (np.log(g_outs) + np.log(fs1) + np.log(fs2)).transpose())))
   # ((g_outs * fs1 * fs2).transpose()*pi)/np.sum(((g_outs * fs1 * fs2).transpose()*pi),1)
#zs =

