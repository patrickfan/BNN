import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  
# import tensorflow as tf


import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import glob
import pandas as pd
from pandas import DataFrame 
import random
from sklearn import metrics
from sklearn.utils import shuffle
import os
import pickle as pkl
import datetime

# ==== fix random seed for reproducibility =====
seed = 1  # use this constant seed everywhere
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(7)
random.seed(1245)
tf.set_random_seed(89)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1, 
                              allow_soft_placement=True,
                              device_count = {'CPU' : 1, 'GPU' : 0})
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

##-----------------------------------------
## --- calculate Prediction interval -----
##-----------------------------------------
def report_on_percentiles(y, y_pred, y_std):
    n = len(y.ravel())

    n1 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 1)
    n2 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 2)
    n3 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 3)
    print('Using {} data points'.format(n))
    print('{} within 1 std'.format(100 * n1 / n))
    print('{} within 2 std'.format(100 * n2 / n))
    print('{} within 3 std'.format(100 * n3 / n))

    return


##- # Puts 1D data into time, lat, lon
def recube(in_array):
    output = np.zeros([time_len, lat_len, lon_len])

    for t in range(time_len):
        output[t,:,:] = in_array[lat_len * lon_len * (t): lat_len * lon_len * (t+1)].reshape([lat_len, lon_len])
    
    return output

def dn(val):    
    return ((1 + val)*(y_max - y_min) / 2) + y_min

def calc_RMSE(y_pred, y):   
    return np.sqrt(np.mean(np.square(y_pred.ravel() - y.ravel())))
                   
def calc_NLL(y_pred, y, y_std):                  
    return np.mean(0.5*((((y_pred.ravel() - y.ravel())**2)/((y_std.ravel()**2)) + np.log(y_std.ravel()**2) + np.log(2*np.pi))))


save_path = 'Result_epoch100/{}'

Obs = np.loadtxt('Z_ERA5-Reanalysis_monthly_precipitation_1980to2014.txt', skiprows=1)
neg_inx = np.argwhere(Obs[:,-1]<0).squeeze()
Obs[neg_inx,5] = 0

time_len = 420
lat_len = 29
lon_len = 61

obs = recube (Obs[:,5])
#With 0.03 bias and 0.005 noise
Model = obs.copy() - 0.03 + np.random.normal(size=[time_len, lat_len, lon_len]) * 0.005

mdl1 = np.random.random( size = [time_len, lat_len, lon_len ]) *5+10
mdl1[:,-15:, :-31] = Model [:,-15:, :-31]
 
mdl2 = np.random.random( size = [time_len, lat_len, lon_len ]) *5+10
mdl2[:,:-15, :-31] = Model[:,:-15, :-31] 
 
mdl3 = np.random.random( size = [time_len, lat_len, lon_len ]) *5+10 
mdl3[:,:-15, -31:] = Model[:,:-15, -31:] 

mdl4 = np.random.random( size = [time_len, lat_len, lon_len ]) *5+10
mdl4[:,-15:, -31:] = Model[:,-15:, -31:] 

# Add noise
# In the NW we have 0.01 noise
obs[:,-15:, :-31] = obs[:,-15:, :-31] + np.random.normal(size=[time_len, 15, lon_len-31 ]) * 0.01
# In the SW we have 0.02 noise
obs[:,:-15, :-31] = obs[:,:-15, :-31] + np.random.normal(size=[time_len, lat_len-15, lon_len-31 ]) * 0.02
# In the SE we have 0.01 noise
obs[:,:-15, -31:] = obs[:,:-15, -31:] + np.random.normal(size=[time_len, lat_len-15, 31]) * 0.02
# In the NE we have 0.02 noise
obs[:,-15:, -31:] = obs[:,-15:, -31:] + np.random.normal(size=[time_len, 15, 31]) * 0.01


Nmodel = 5  # num_simulation + obs
Ndata = 742980
Ntrain = lat_len*lon_len*12*20

df = pd.DataFrame()
df['mons'] = Obs[:,0]+1
df['mon_num'] = Obs[:,2]
df['lat'] = Obs[:,3]
df['lon'] = Obs[:,4]+360
df['Obs'] = obs.ravel()

df['mdl1'] = mdl1.ravel()
df['mdl2'] = mdl2.ravel()
df['mdl3'] = mdl3.ravel()
df['mdl4'] = mdl4.ravel()

models = df[df.columns[4:]]

ModAve = df.iloc[:,5:].mean(axis=1)
print(ModAve.shape, ModAve.min(), ModAve.max())
models['ModAve'] = ModAve

# Apply coordinate mapping lat,lon -> x,y,z
lon = df['lon'] * np.pi / 180
lat = df['lat'] * np.pi / 180
x = np.cos(lat) * np.cos(lon)
y = np.cos(lat) * np.sin(lon)
z = np.sin(lat)

# Apply coordinate mapping month_number -> x_mon, y_mon
rads = (df['mon_num'] * 360/12) * (np.pi / 180)
x_mon = np.sin(rads)
y_mon = np.cos(rads)

# min-max scale months (months since Jan 1980)
mons_scaled = 2 * (df['mons'] - df['mons'].min())/(df['mons'].max() - df['mons'].min()) - 1

# Remove old coords and add new mapped coords from/to dataframe
df = df.drop(['lat', 'lon', 'mon_num', 'mons'], axis=1)
df['x'] = x
df['y'] = y
df['z'] = z
df['x_mon'] = x_mon
df['y_mon'] = y_mon
df['mons'] = mons_scaled

# Apply min-max scaling to each model and observations
y_min = df['Obs'].min()
y_max = df['Obs'].max()

print('df columns',df.columns)

for i in np.arange(Nmodel): # 14 models and 1 obs
    mdl = df[df.columns[i]]
    df[df.columns[i]] = 2 * (mdl - mdl.min())/(mdl.max() - mdl.min()) - 1
    
# Apply coordinate scaling
df['x'] = df['x'] * 2
df['y'] = df['y'] * 2
df['z'] = df['z'] * 2

df['x_mon'] = df['x_mon'] * 1.0
df['y_mon'] = df['y_mon'] * 1.0
df['mons'] = df['mons'] * 1.0

##--- Create training and testing dataset
df_train = df.iloc[:Ntrain,:]
df_test = df.iloc[Ntrain:,:]

# In sample training
X_train = df_train.drop(['Obs'],axis=1).values
y_train = df_train['Obs'].values.reshape(-1,1)

# The in sample testing - this is not used for training
X_test = df_test.drop(['Obs'],axis=1).values
y_test = df_test['Obs'].values.reshape(-1,1)

# For all time
X_at = df.drop(['Obs'],axis=1).values
y_at = df['Obs'].values.reshape(-1,1)

print('#Training data X and y size', np.shape(X_train), np.shape(y_train))
print('#Testing data X and y size', np.shape(X_test), np.shape(y_test))
print('#All the data X and y size', np.shape(X_at), np.shape(y_at))

num_models = Nmodel-1

# prior on the noise 
noise_mean = 0.015
noise_std = 0.001

# hyperparameters
n = X_train.shape[0]
x_dim = X_train.shape[1]
alpha_dim = x_dim - num_models
y_dim = y_train.shape[1]

n_ensembles = 50
hidden_size = 100

# Factor to account for 1st layer std being below 1

init_stddev_1_w =  np.sqrt(3.0/(alpha_dim)) # tune the coefficient to ensure layer 1 mean=0, layer 1 std=1.0
init_stddev_1_b = init_stddev_1_w
init_stddev_2_w =  np.sqrt(0.3/hidden_size) # tune the coefficient to ensure layer 1 mean=0, layer 1 std=1.0
init_stddev_2_b = init_stddev_2_w
init_stddev_noise_w = (0.01*noise_std)/np.sqrt(hidden_size)

lambda_anchor = 1.0/(np.array([init_stddev_1_w,init_stddev_1_b,init_stddev_2_w,init_stddev_2_b,init_stddev_noise_w])**2)

n_epochs = 6000
batch_size = 20000
learning_rate = 0.00005

# NN class
class NN():
    def __init__(self, x_dim, y_dim, hidden_size, init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, init_stddev_2_b, init_stddev_noise_w, learning_rate):
        # setting up as for a usual NN
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # set up NN
        self.inputs = tf.placeholder(tf.float32, [None, x_dim], name='inputs')
        self.modelpred = self.inputs[:, :num_models]
        self.spacetime = self.inputs[:, num_models: num_models + alpha_dim]
        self.area_weights = self.inputs[:, -1]
        self.y_target = tf.placeholder(tf.float32, [None, y_dim], name='target')
        
        self.layer_1_w = tf.layers.Dense(hidden_size, activation=tf.nn.tanh,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_w),
                                         bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_b))
        self.layer_1 = self.layer_1_w.apply(self.spacetime)
        self.layer_2_w = tf.layers.Dense(num_models, activation=None,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_2_w),
                                         bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_2_b))
        self.layer_2 = self.layer_2_w.apply(self.layer_1)

        self.model_coeff = tf.nn.softmax(self.layer_2)

            
        self.output = tf.reduce_sum(self.model_coeff * self.modelpred, axis=1) 
        
        self.noise_w = tf.layers.Dense(self.y_dim, activation=None, use_bias=False,
                                       kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_noise_w))
        self.noise_pred = self.noise_w.apply(self.layer_1)

        # set up loss and optimiser - we'll modify this later with anchoring regularisation
        self.opt_method = tf.train.AdamOptimizer(self.learning_rate)
        self.noise_sq = tf.square(self.noise_pred + noise_mean)[:,0] + 1e-6 
        self.err_sq = tf.reshape(tf.square(self.y_target[:,0] - self.output), [-1])
        num_data_inv = tf.cast(tf.divide(1, tf.shape(self.inputs)[0]), dtype=tf.float32)

        self.mse_ = num_data_inv * tf.reduce_sum(self.err_sq) 
        self.loss_ = num_data_inv * (tf.reduce_sum(tf.divide(self.err_sq, self.noise_sq)) + tf.reduce_sum(tf.log(self.noise_sq)))
        self.optimizer = self.opt_method.minimize(self.loss_)

        return

    def get_weights(self, sess):
        '''method to return current params'''
        ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.layer_2_w.kernel, self.layer_2_w.bias,  self.noise_w.kernel]
        w1, b1, w2, b2,  wn = sess.run(ops)
        return w1, b1, w2, b2,  wn

    def anchor(self, sess, lambda_anchor):
        '''regularise around initialised parameters'''
        w1, b1, w2, b2, wn = self.get_weights(sess)

        # get initial params
        self.w1_init, self.b1_init, self.w2_init, self.b2_init,  self.wn_init = w1, b1, w2, b2, wn
        loss_anchor = lambda_anchor[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
        loss_anchor += lambda_anchor[1]*tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
        loss_anchor += lambda_anchor[2]*tf.reduce_sum(tf.square(self.w2_init - self.layer_2_w.kernel))
        loss_anchor += lambda_anchor[3]*tf.reduce_sum(tf.square(self.b2_init - self.layer_2_w.bias))
        
        loss_anchor += lambda_anchor[4]*tf.reduce_sum(tf.square(self.wn_init - self.noise_w.kernel)) # new param

        self.loss_anchor = tf.cast(1.0/X_train.shape[0], dtype=tf.float32) * loss_anchor
        
        # combine with original loss
        self.loss_ = self.loss_ + tf.cast(1.0/X_train.shape[0], dtype=tf.float32) * loss_anchor
        self.optimizer = self.opt_method.minimize(self.loss_)
        return

    def predict(self, x, sess):
        '''predict method'''
        feed = {self.inputs: x}
        y_pred = sess.run(self.output, feed_dict=feed)
        return y_pred
    
    def get_noise_sq(self, x, sess):
        '''get noise squared method'''
        feed = {self.inputs: x}
        noise_sq = sess.run(self.noise_sq, feed_dict=feed)
        return noise_sq

    def get_alphas(self, x, sess):
        feed = {self.inputs: x}
        alpha = sess.run(self.model_coeff, feed_dict=feed)
        return alpha

    def get_alpha_w(self, x, sess):
        feed = {self.inputs: x}
        alpha_w = sess.run(self.layer_2, feed_dict=feed)
        return alpha_w

    def get_w1(self, x, sess):
        feed = {self.inputs: x}
        w1 = sess.run(self.layer_1, feed_dict=feed)
        return w1

def fn_predict_ensemble(NNs,X_train):
    y_pred=[]
    y_pred_noise_sq=[]
    for ens in range(0,n_ensembles):
        y_pred.append(NNs[ens].predict(X_train, sess))
        y_pred_noise_sq.append(NNs[ens].get_noise_sq(X_train, sess))
    y_preds_train = np.array(y_pred)
    y_preds_noisesq_train = np.array(y_pred_noise_sq)
    y_preds_mu_train = np.mean(y_preds_train,axis=0)
    y_preds_std_train_epi = np.std(y_preds_train,axis=0)
    y_preds_std_train = np.sqrt(np.mean((y_preds_noisesq_train + np.square(y_preds_train)), axis = 0) - np.square(y_preds_mu_train)) #add predicted aleatoric noise
    return y_preds_train, y_preds_mu_train, y_preds_std_train, y_preds_std_train_epi, y_preds_noisesq_train

def get_alphas(NNs, X_train):
    alphas = []
    for ens in range(0,n_ensembles):
        alphas.append(NNs[ens].get_alphas(X_train, sess))
    return alphas


def get_layer2_output(NNs, X_train):
    alpha_w = []
    for ens in range(0,n_ensembles):
        alpha_w.append(NNs[ens].get_alpha_w(X_train, sess))
    return alpha_w

def get_layer1_output(NNs, X_train):
    w1 = []
    for ens in range(0,n_ensembles):
        w1.append(NNs[ens].get_w1(X_train, sess))
    return w1

def get_w1(NNs, X_train):
    w1 = []
    for ens in range(0,n_ensembles):
        w1.append(NNs[ens].get_w1(X_train, sess))
    return w1

def get_alpha_w(NNs, X_train):
    alpha_w = []
    for ens in range(0,n_ensembles):
        alpha_w.append(NNs[ens].get_alpha_w(X_train, sess))
    return alpha_w

##----------------------------------
## ------ Initialise the NNs -------
##----------------------------------
NNs=[]

# sess = tf.Session()

init_weights = []

# loop to initialise all ensemble members
for ens in range(0,n_ensembles):
    NNs.append(NN(x_dim, y_dim, hidden_size, init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, init_stddev_2_b, init_stddev_noise_w,learning_rate))
    # initialise only unitialized variables - stops overwriting ensembles already created
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
      
    # do regularisation now that we've created initialisations
    NNs[ens].anchor(sess, lambda_anchor)


##------------------------------------------------------
## --- Check the Priors  -----
##------------------------------------------------------
## the number of data points used to check the priors=10000here * n_ensembles * num_models
Nsample = lat_len*lon_len*5

X_train_short = X_train[:Nsample]

y_preds_train, y_preds_mu_train, y_preds_std_train, y_preds_std_train_epi, y_preds_noisesq_train = fn_predict_ensemble(NNs,X_train_short)
plt.figure(figsize=(8,8))
plt.errorbar(y_train[:Nsample], y_preds_mu_train[:Nsample], yerr=(y_preds_std_train*1)[:Nsample],linewidth = 0.25, color = 'gray', ms=2,mfc='red',mec='black', fmt='o')
plt.plot(np.arange(np.min(y_train[:Nsample]), np.max(y_train[:Nsample]), 0.01), np.arange(np.min(y_train[:Nsample]), np.max(y_train[:Nsample]), 0.01), linewidth = 2, linestyle = 'dashed',zorder = 100)
plt.xlabel('True concentration')
plt.ylabel('Predicted concentration')
plt.savefig('Prior_predictive_check.png')



y_preds_train, y_preds_mu_train, y_preds_std_train, y_preds_std_train_epi, y_preds_noisesq_train = fn_predict_ensemble(NNs,X_train)

# Alphas
alphas = np.array(get_alphas(NNs, X_train_short))
print('Alpha mean should be: {}'.format(1/num_models))
print('Alpha mean is: {}'.format(np.mean(np.array(alphas).ravel())))
print('Alpha std should be: {}'.format(np.sqrt((1/(1 + num_models)) * (1/num_models)*(1-(1/num_models)))))
print('Alpha std is: {}'.format(np.mean(np.std(np.array(alphas), axis=0).ravel())))
report_on_percentiles(alphas, np.array(1/num_models), np.mean(np.std(np.array(alphas), axis=0).ravel()))
print('')


### Network weights
print('For the layers')
w1 = np.array(get_w1(NNs, X_train_short))
alpha_w = np.array(get_alpha_w(NNs, X_train_short))
print('Layer 1 mean: {}'.format(np.mean(w1.ravel())))
print('Layer 2 mean: {}'.format(np.mean(alpha_w.ravel())))
print('Layer 1 Std: {}'.format(np.mean(np.std(w1, axis=0).ravel())))
print('Layer 2 Std: {}'.format(np.mean(np.std(alpha_w, axis=0).ravel())))
print('')

### Noise
print('For noise')
pred_noise = np.sqrt(np.array([NN.get_noise_sq(X_train, sess) for NN in NNs]))
print('Mean noise is: {}'.format(np.mean(pred_noise)))
print('Mean noise should be: {}'.format(noise_mean))
print('Std noise is: {}'.format(np.std(pred_noise)))
print('Std noise should be: {}'.format(noise_std))

report_on_percentiles(pred_noise, np.array(noise_mean), np.array(noise_std))
print('')

##----------------------------------
## -----  Training  ----------
##----------------------------------
l_s = []
m_s = []
a_s = []
saver = tf.train.Saver()

for ens in range(0,n_ensembles):
    ep_ = 0
    losses = []
    mses = []
    anchs = []
    print('**************** NN:',ens + 1)
    while ep_ < n_epochs:
        if (ep_ % 50 == 0):
            X_train, y_train = shuffle(X_train, y_train, random_state = ep_)

        ep_ += 1
        for j in range(int(n/batch_size)): #minibatch training loop
            feed_b = {}
            feed_b[NNs[ens].inputs] = X_train[j*batch_size:(j+1)*batch_size, :]
            feed_b[NNs[ens].y_target] = y_train[j*batch_size:(j+1)*batch_size, :]
            blank = sess.run(NNs[ens].optimizer, feed_dict=feed_b)
        if (ep_ % 1) == 0: 
            feed_b = {}
            feed_b[NNs[ens].inputs] = X_train
            feed_b[NNs[ens].y_target] = y_train
            loss_mse = sess.run(NNs[ens].mse_, feed_dict=feed_b)
            loss_anch = sess.run(NNs[ens].loss_, feed_dict=feed_b)
            loss_anch_term = sess.run(NNs[ens].loss_anchor, feed_dict=feed_b)
            losses.append(loss_anch)
            mses.append(loss_mse)
            anchs.append(loss_anch_term)
        if (ep_ % 10 == 0):
            print('epoch:' + str(ep_) + ' at ' + str(datetime.datetime.now()))
            print('--- rmse_', np.round(np.sqrt(loss_mse),5), ', loss_anch', np.round(loss_anch,5), ', anch_term', np.round(loss_anch_term,5))
        # If saving weights
        if (ep_ % 500 == 0):      
            weight = NNs[ens].get_weights(sess)
            pkl.dump(weight, open(save_path.format('weights{}_{}.pkl'.format(ens,ep_)), 'wb'))

    l_s.append(losses)
    m_s.append(mses)
    a_s.append(anchs)


np.savetxt('Anchor_loss.dat',np.array(l_s).T)
np.savetxt('MSE_loss.dat',np.array(m_s).T)
np.savetxt('AnchorTerm.dat',np.array(a_s).T)

fig = plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.plot(np.array(l_s).T)
plt.title('Anchored loss')
plt.subplot(1,3,2)
plt.plot(np.array(m_s).T)
plt.yscale('log')
plt.title('MSE')
plt.subplot(1,3,3)
plt.plot(np.array(a_s).T)
plt.title('Anchoring term')
fig.tight_layout()
plt.savefig('Loss.png')

# exit()

print ((np.array(l_s).T).shape)

y_preds_train, y_preds_mu_train, y_preds_std_train,  y_preds_std_train_epi, y_preds_noisesq_train = fn_predict_ensemble(NNs,X_train)
y_preds_test, y_preds_mu_test, y_preds_std_test,  y_preds_std_test_epi, y_preds_noisesq_test = fn_predict_ensemble(NNs,X_test)
y_preds_at, y_preds_mu_at, y_preds_std_at,  y_preds_std_at_epi, y_preds_noisesq_at= fn_predict_ensemble(NNs,X_at)


## -- scale it back to original space
y_preds_mu_train = dn(y_preds_mu_train)
y_preds_mu_test = dn(y_preds_mu_test) 
y_preds_mu_at = dn(y_preds_mu_at)
y_train = dn(y_train)
y_test = dn(y_test)
y_at = dn(y_at)
y_preds_std_train = dn(y_preds_std_train)
y_preds_std_test = dn(y_preds_std_test)
y_preds_std_at = dn(y_preds_std_at)

np.savetxt('BNN_pred.out',y_preds_mu_at)

## -- Look at RMSEs and NLLs
print('Train RMSE: {}'.format(calc_RMSE(y_preds_mu_train, y_train)))
print('Test RMSE: {}'.format(calc_RMSE(y_preds_mu_test, y_test)))
print('Total RMSE: {}'.format(calc_RMSE(y_preds_mu_at, y_at)))

print('Train NLL: {}'.format(calc_NLL(y_preds_mu_train, y_train, y_preds_std_train)))
print('Test NLL: {}'.format(calc_NLL(y_preds_mu_test, y_test, y_preds_std_test)))
print('All NLL: {}'.format(calc_NLL(y_preds_mu_at, y_at, y_preds_std_at)))

print('shape of BNN_train and BNN_test', y_preds_mu_train.shape, y_preds_mu_test.shape)
print('min and max of BNN_train: ', y_preds_mu_train.min(), y_preds_mu_train.max())
print('min and max of BNN_test: ', y_preds_mu_test.min(), y_preds_mu_test.max())
print('min and max of obs: ', y_at.min(), y_at.max())

fig = plt.figure(figsize=(5,4))
sns.distplot(y_preds_mu_at, hist=False, kde=True, label='BNN average',color='blue', kde_kws={'linestyle':'--'})
sns.distplot(ModAve, hist=False, kde=True, label='Simple average',color='green', kde_kws={'linestyle':'-.'})
sns.distplot(y_at, hist=False, kde=True, label='Obs',color='red', kde_kws={'linestyle':'-'})
plt.legend(prop={'size': 10})
plt.xlabel('Precip (monthly)')
plt.ylabel('Density')  
plt.xlim([0,16])
plt.savefig('Hist_precip.png')
#plt.show()


print(' -------- For train -------- ')
report_on_percentiles(y_train, y_preds_mu_train, y_preds_std_train)
print(' -------- For test -------- ')
report_on_percentiles(y_test, y_preds_mu_test, y_preds_std_test)

## -------------------------------------
# Model Coefficients-- alpha
## -------------------------------------
alphas = np.array(get_alphas(NNs, X_at))
alpha = np.mean(alphas, axis=0)
print('Shape of alphas and alpha: ', np.shape(alphas), np.shape(alpha))

##-- save alpha [#data, #model]
pkl.dump(alpha, open('alpha.pkl', 'wb'))

fig = plt.figure(figsize=(18,6))
for i in range(num_models):
    a = alphas[:,:,i]
    plt.subplot(2,2,i + 1)
    #plt.plot([0,Ntime], [1/num_models,1/num_models], '--', color='black')
    plt.plot(np.mean(recube(np.mean(a, axis=0)), axis=(1,2)))
    #print ("model", i, np.mean(recube(np.mean(a, axis=0)), axis=(1,2)))
    plt.title(df.columns[i+1])
    #plt.ylim([0,0.4])

    plt.xlabel('Months in 1980-2014')

    plt.ylabel('Model weight')
fig.tight_layout()
plt.savefig('Mean_Alphas.png')
#exit()


## -------------------------------------
# Noise predictions
## -------------------------------------
aletoric_noise = []

for NN in NNs:
    feed_b = {}
    feed_b[NN.inputs] = X_at
    feed_b[NN.y_target] = y_at
    noise_sq = sess.run(NN.noise_sq, feed_dict=feed_b)
    aletoric_noise.append(noise_sq)

a_n = recube(np.sqrt(np.mean(np.array(aletoric_noise), axis=0)))

## - save noise a_n[#data,1]
pkl.dump(a_n, open('aleatoric_noise.pkl', 'wb'))

epi = recube(y_preds_std_at_epi)
pkl.dump(epi, open('epi.pkl', 'wb'))
print (epi.shape)


saver.save(sess,'net/my_net.ckpt')
