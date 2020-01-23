# %% IMPORT SECTION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import time
import sklearn.metrics as metrics
import psutil
from multiprocessing import Pool

from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, classification_report
from collections import Counter
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Embedding
from keras import backend as K
from keras.layers import LSTM, Dense, Bidirectional, Input, Dropout, BatchNormalization, CuDNNGRU, CuDNNLSTM
from keras.layers import Flatten, TimeDistributed, Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.regularizers import l1_l2, l1, l2
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
from keras.constraints import unit_norm
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")
colors = np.array(['#0101DF', '#DF0101', '#01DF01'])

time0 = time.time()
rs = 42

def compute_metrics(y_te2, y_p, string):
    print('\n %s' % string)
    print('CONFUSION MATRIX')
    print(metrics.confusion_matrix(y_te2, y_p))
    print(classification_report(y_te2, y_p))
    print('\nROC CURVE: %2.2f' % roc_auc_score(y_te2, y_p))

def plotting_ROC_curve(X_te, y_te, model):
    plt.figure()
    metrics.plot_roc_curve(model, X_te, y_te)
    plt.plot([0, 1], [0, 1], label='baseline', c='red')
    plt.legend(loc='lower right')
    plt.show()


def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


def Ratio10(y, string):
    suma = Counter(y)[0] + Counter(y)[1]
    print('\n' + string)
    print(Counter(y))
    print('0: %3.2f%%\n1: %3.2f%%' % (100 * Counter(y)[0] / suma, 100 * Counter(y)[1] / suma))


def histo(data, var):
    fig = plt.figure()
    sns.countplot(var, data=data, palette=colors)
    plt.title('0/1 Distribution (0: No Stalling || 1: Stalling Event', fontsize=14)
    plt.show()


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def build_model(n_layers, input_dim, units, activation, initializer,regu):
    if isinstance(units, list):
        assert len(units) == n_layers
    else:
        units = [units] * n_layers

    model = Sequential()
    # Adds first hidden layer with input_dim parameter
    model.add(Dense(units=units[0],
                    input_dim=input_dim,
                    activation=activation,
                    kernel_initializer=initializer,
                    bias_regularizer=l2(regu),
                    name='h1'))
    model.add(Dropout(0.25))
    # Adds remaining hidden layers
    for i in range(2, n_layers + 1):
        model.add(Dense(units=units[i - 1],
                        activation=activation,
                        kernel_initializer=initializer,
                        bias_regularizer=l2(regu),
                        name='h{}'.format(i)))
    model.add(Dropout(0.5))
    # Adds output layer
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=initializer, name='o'))
    return model


# %% LOAD DATASET REMOVING NOISE

dataset = pd.read_csv('../../output/df_general.csv', sep=',')
split = 0.8
del dataset['No.']
del dataset['Protocol']
del dataset['tcp_flag_cwr']
del dataset['tcp_flag_ecn']
del dataset['tcp_flag_urg']
del dataset['ip_len']
del dataset['tcp_flag_psh']
del dataset['tcp_flag_fin']
del dataset['prebuffering']
del dataset['Time']

dataset = dataset.fillna(dataset.mean())

print('Original Dataset')
print(dataset.columns)
Ratio10(dataset['stalling_event'], 'dataset[stalling_event]')
#histo(dataset, 'stalling_event')

# %%

id = list(dataset[dataset['∆t'] > 0.5].index)
dataset = dataset.drop(id)

#%% SMOTE
'''y = dataset['stalling_event']
X = dataset.drop(['stalling_event','capture','delay'],axis=1)
print(X.columns)


from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority',random_state=rs,n_jobs=-1)
X, y = smote.fit_sample(X.values, y.values)
Ratio10(y,'y (SMOTE)')
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)

# %%
y_test = list(map(int, y_test))
y_train = list(map(int, y_train))
Ratio10(y_test, 'y_test')
Ratio10(y_train, 'y_train')

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority',random_state=rs,n_jobs=-1)
X_train, y_train = smote.fit_sample(X_train, y_train)
Ratio10(y_train,'y_train (SMOTE)')
X_test, y_test = smote.fit_sample(X_test, y_test)
Ratio10(y_test,'y_test (SMOTE)')


mms = preprocessing.MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)
# %%

'''X_train = pd.DataFrame(data=X_train, columns=dataset.columns[:-3])
X_test = pd.DataFrame(data=X_test, columns=dataset.columns[:-3])

y_train = pd.DataFrame(data=y_train, columns=['y_train'])
y_test = pd.DataFrame(data=y_test, columns=['y_test'])'''

# %%
n_features = X_train.shape[1]
'''
for i in range(1,4): # number of layers
    for j in [4,8,10,12,20]:
        for k in ['he_normal','orthogonal','glorot_uniform']:
            for z in [16,32,64]:
                print('\nPARAMETERS\n'
                      'n_inputs: %i\n'
                      'n_layers: %i\n'
                      'nodes/layer: %i\n'
                      'weights_init: %s\n'
                      'batch_size: %i\n'
                      %(n_features,i,j,k,z))
                model = build_model(i, n_features, j, 'relu', k)
                print(model.summary())
                print('Built NN...')
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
                print('Compiled NN...')
                history = model.fit(X_train, y_train, epochs=10, batch_size=z, verbose=1, validation_split=0.2)
                print('Fitted NN...')
                train_loss, train_acc = model.evaluate(X_train, y_train, verbose=1)
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
                print('Train Accuracy: %.3f | Test Accuracy: %.3f' %(train_acc,test_acc))
                #y_pred = model.predict(X_test)
                y_pred = model.predict_classes(X_test)
                #Ratio10(y_test,'y_test')
                print(y_pred.T.ravel())
                Ratio10(y_pred.T.ravel(),'y_pred')
                compute_metrics(y_test,y_pred.T.ravel(),'Neural Network testing...')
                plt.title('PARAMETERS\n'
                      'n_inputs: %i'
                      'n_layers: %i'
                      'nodes/layer: %i'
                      'weights_init: %s'
                      'batch_size %i'
                      %(n_features,i,j,k,z))
                plt.subplot(211)
                plt.title('Loss')
                plt.plot(history.history['loss'], label='train')
                plt.plot(history.history['val_loss'], label='test')
                plt.legend()
                plt.subplot(212)
                plt.title('ACCURACY')
                plt.plot(history.history['accuracy'], label='train')
                plt.plot(history.history['val_accuracy'], label='test')
                plt.legend()
                plt.savefig('../output/PARAMETERS-%i%i%i%s%i.png'%(n_features,i,j,k,z))
                plt.show()'''
i = 4
j = [128,128,64,16]
k = 'orthogonal'
z = 32
ep = 128
regula = 0.0001
print('\nPARAMETERS\n'
      'n_inputs: %i\n'
      'n_layers: %i\n'
      'epochs: %i\n'
      'weights_init: %s\n'
      'batch_size: %i\n'
      'regularized: %f\n'
      %(n_features,i,ep,k,z,regula))
model = build_model(i, n_features, j, 'relu', k,regula)
print(model.summary())
print('Built NN...')
model.compile(optimizer='adam', loss=f1_loss, metrics=['binary_accuracy',f1])
print('Compiled NN...')
history = model.fit(X_train, y_train, epochs=ep, batch_size=z, verbose=1, validation_split=0.1)
print('Fitted NN...')
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=1)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print('Train Accuracy: %.3f | Test Accuracy: %.3f' %(train_acc,test_acc))
#y_pred = model.predict(X_test)
y_pred = model.predict_classes(X_test)
#Ratio10(y_test,'y_test')
print(y_pred.T.ravel())
Ratio10(y_pred.T.ravel(),'y_pred')
compute_metrics(y_test,y_pred.T.ravel(),'Neural Network testing...')
plt.title('PARAMETERS\n'
      'n_inputs: %i'
      'n_layers: %i'
      'nodes/layer: %i'
      'weights_init: %s'
      'batch_size %i'
      %(n_features,i,j,k,z))
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.subplot(212)
plt.title('ACCURACY')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.savefig('../output/PARAMETERS-%i%i%i%s%i.png'%(n_features,i,j,k,z))
plt.show()