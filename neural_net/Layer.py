import tensorflow as tf
import numpy as np

class Layer():

    def __init__(self, neural_net, layer_index, input_variables, input_size, output_size, time_lags):
        self.neural_net = neural_net
        self.activation_function = neural_net.get_activation_function(layer_index)
        self.layer_index = layer_index
        self.input_variables = input_variables
        self.input_size = input_size
        self.output_size = output_size
        self.name = "Layer-"+str(self.layer_index)
        self.time_lags = time_lags
        #self.build()


    def build(self):
        start_range_weights = self.neural_net.get_initial_weight_range()[0]
        end_range_weights = self.neural_net.get_initial_weight_range()[1]
        self.weights = tf.Variable(np.random.uniform(start_range_weights, end_range_weights, size=(self.input_size, self.output_size)),
                                   name=self.name + '-weigths', trainable=True)
        start_range_bias_weights = self.neural_net.get_initial_bias_weigh_range()[0]
        end_range_bias_weights = self.neural_net.get_initial_bias_weigh_range()[1]
        self.biases = tf.Variable(np.random.uniform(start_range_bias_weights, end_range_bias_weights, size=self.output_size),
                                  name=self.name + '-bias', trainable=True)
        if (self.activation_function == "relu"):
            self.output_variables = tf.nn.relu(tf.matmul(self.input_variables, self.weights) + self.biases, name=self.name + '-output')

        elif (self.activation_function == "sigmoid"):
            self.output_variables = tf.nn.sigmoid(tf.matmul(self.input_variables, self.weights) + self.biases, name=self.name + '-output')

        elif (self.activation_function == "tanh"):
            self.output_variables = tf.nn.tanh(tf.matmul(self.input_variables, self.weights) + self.biases, name=self.name + '-output')

        else:
            raise ValueError('Activation function does not exist')


    def get_layer_variables(self,type):  # type = (in,out,wgt,bias)
        return {'input': self.input_variables, 'output': self.output_variables, 'weigth': self.weights, 'bias': self.biases}[type]

    def get_act_function(self):
        return self.activation_function

    def get_output_size(self):
        return self.output_size

#type - which of variable, ex "weights", "output", "input", "bias"
#spec - which visualization do you want, ex "avg", "hist", "max", "min"
    def gen_probe(self,type,spec):
        var = self.get_layer_variables(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)
