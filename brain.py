from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD, RMSprop
from keras.callbacks import Callback, ModelCheckpoint
import tensorflow as tf

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)

def huber_loss_simple(y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)


class Brain():
    def __init__(self, learning_rate, input_shape, action_space = 3, model_print=True):
        
        self.input_shape = input_shape
        self.action_space = action_space
        
        # The model
        #self.model = Sequential()
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(action_space, activation='softmax'))

        optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01)
        self.model.compile(loss=huber_loss_simple, optimizer=optimizer) #mse
        
        if model_print:
            self.model.summary()
            
        # Checkpoint & Loss History
        #self.checkpointer = ModelCheckpoint(filepath='models/weights.hdf5', verbose=0, save_best_only=True)
        self.history = LossHistory()
    
    def save_model(self, filename):
        self.model.save(filename)
    
    def load_model(self, model_filename):
        self.model = load_model(model_filename)
        
    def train(self, X_train, y_train, batch_size=32, epochs = 1):        
        self.model.fit(X_train, y_train, batch_size=batch_size, 
                       epochs=epochs, verbose=0, validation_split = 0.3, callbacks=[self.history])#self.checkpointer, self.history])
    
    def predict(self, state):
        return self.model.predict(state)#.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])) #input is in shape (batch, 125, 80)