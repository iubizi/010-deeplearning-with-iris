####################
# Disable full memory lock
####################

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

####################
# load iris
####################

from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

####################
# onehot encoding
####################

from tensorflow.keras.utils import to_categorical

y_onehot = to_categorical(y)

####################
# train test split
####################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y_onehot, test_size=0.2, random_state=42, stratify=y)

####################
# dl model
####################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def get_model():
    
    model = Sequential()
    
    model.add(Dense(16, activation = 'relu', input_shape = (4,)))
    # model.add(Dropout(0.5))
    model.add(Dense(16, activation = 'relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(3, activation = 'softmax'))

    return model

####################
# model build
####################

model = get_model()

from tensorflow.keras.optimizers import Adam

model.compile( optimizer = Adam( lr = 1e-4 ),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'] )

model.summary()

####################
# Early Stopping
####################

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor = 'val_accuracy',
    patience = 20,
    restore_best_weights = True )

####################
# train and val
####################

model.fit( x_train, y_train,
           validation_data = (x_test, y_test),

           epochs = 100, batch_size = 4,

           callbacks = [early_stopping],

           verbose = 2, # Disable progress bar
           shuffle = True,
           )
