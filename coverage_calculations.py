import numpy as np
from keras import models
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from PIL import Image

# NEURON COVERAGE model from https://github.com/geifmany/cifar-vgg

class NeuronCoverage:
    """
    Implements Neuron Coverage metric from "DeepXplore: Automated Whitebox Testing of Deep Learning Systems" by Pei
    et al.
    Supports incremental measurements using which one can observe the effect of new inputs to the coverage
    values.
    """

    def __init__(self, model, scaler=default_scale, threshold=0, skip_layers=None, predictions=None):
        self.activation_table = defaultdict(bool)

        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        self.predictions = predictions


    def get_measure_state(self):
        return [self.activation_table]

    def set_measure_state(self, state):
        self.activation_table = state[0]

    def test(self, test_inputs):
        """
        :param test_inputs: Inputs
        :return: Tuple containing the coverage and the measurements used to compute the coverage. 0th element is the
        percentage neuron coverage value.
        """
        outs = []

        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
        used_inps = []
        nc_cnt = 0
        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            inp_cnt = 0
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    if not self.activation_table[(layer_index, neuron_index)] and np.mean(
                            out_for_input[..., neuron_index]) > self.threshold and inp_cnt not in used_inps:
                        used_inps.append(inp_cnt)
                        nc_cnt += 1
                    self.activation_table[(layer_index, neuron_index)] = self.activation_table[
                                                                             (layer_index, neuron_index)] or np.mean(
                        out_for_input[..., neuron_index]) > self.threshold

                inp_cnt += 1

        covered = len([1 for c in self.activation_table.values() if c])
        covered_arr = []
        for c in self.activation_table.values():
          if c:
            covered_arr.append(1)
          else:
            covered_arr.append(0)
        total = len(self.activation_table.keys())

        return percent_str(covered, total), covered, total, outs, nc_cnt, covered_arr
    

# https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py

class cifar10vgg:
    def __init__(self,train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
       
        self.model.load_weights('/content/drive/MyDrive/thesis/cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


def percent_str(part, whole):
    return "{0}%".format(float(part) / whole * 100)

def get_layer_outs_new(model, inputs, skip=[]):
    evaluater = models.Model(inputs=model.input,
                             outputs=[layer.output for index, layer in enumerate(model.layers) \
                                      if index not in skip])
    return evaluater.predict(inputs, batch_size=32)


def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin

    return X_scaled


# LOAD THE MODEL

def get_trainable_layers(model):
    trainable_layers = []
    for idx, layer in enumerate(model.layers):
        try:
            if 'input' not in layer.name and 'softmax' not in layer.name and \
                    'pred' not in layer.name and 'drop' not in layer.name:
                weights = layer.get_weights()[0]
                trainable_layers.append(model.layers.index(layer))
        except:
            pass

    # trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return trainable_layers


model = cifar10vgg().model


# 2) Load necessary information
trainable_layers = get_trainable_layers(model)
non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
print('Trainable layers: ' + str(trainable_layers))
print('Non trainable layers: ' + str(non_trainable_layers))

experiment_folder = 'experiments'

#Investigate the penultimate layer
subject_layer = -1
subject_layer = trainable_layers[subject_layer]

skip_layers = [0] #SKIP LAYERS FOR NC, KMNC, NBC etc.
for idx, lyr in enumerate(model.layers):
  if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

print("Skipping layers:", skip_layers)

# LOAD THE IMAGES TO AN ARRAY


max_range_start = 1 # number of images to process per batch start index
max_range_end = 251 # number of images to process per batch end index
location_to_save_images = "/sample/location/to/dataset"
generated_images = []
for i in range(max_range_start,max_range_end):
  im = Image.open(location_to_save_images+str(i)+".jpeg")
  y_prob = model.predict(np.expand_dims(im, axis=0))
  a = np.expand_dims(im, axis=0)
  a = np.squeeze(a, axis=0)
  generated_images.append(a)


# CALCULATE COVERAGE

predictions = get_layer_outs_new(model, np.array(generated_images))

nc = NeuronCoverage(model, threshold=.25, skip_layers = skip_layers, predictions=predictions)
coverage, covered1, total, _, _, coverage_arr = nc.test(np.array(generated_images))
print("Your test set's coverage for 25 threshold is: ", coverage)

nc = NeuronCoverage(model, threshold=.65, skip_layers = skip_layers, predictions=predictions)
coverage, covered1, total, _, _, coverage_arr = nc.test(np.array(generated_images))
print("Your test set's coverage for 65 threshold is: ", coverage)