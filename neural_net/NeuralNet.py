import tensorflow as tf
import numpy as np
import CaseManager as cm
import Layer as l
import FlowTools as flt
import matplotlib.pyplot as PLT
import math



class NeuralNet():


#layerDimensions = [input size, hidden layer1 size, hidden layer 2 size, ..., output layer size]
#activationFunctions = [act function to h1, act function to h2, ...., act function to output] write them as string
#initial_weight_range = [start_range, stop_range], ex [-1.0, 1.0]
#cases = [[input1, input2, input3, ...], [target1, target2, ...]]
#time_lags is number of steps in previous states in the reccurent network
#learning_method as a string, example "gradient_decent"
#number_of_target_possibilities is [down, same, up] or [-inf to -1, -1 to 1, 1 to 5, 5 to inf]
#validation_interval is how often

    def __init__(self, layer_dimensions, activation_functions, learning_rate, minibatch_size,
                 initial_weight_range, initial_bias_weight_range, time_lags, cost_function,
                 learning_method, case_manager, validation_interval=None, show_interval=None, softmax=True):
        self.layer_dimensions = layer_dimensions
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.activation_functions = activation_functions
        self.initial_weight_range = initial_weight_range
        self.initial_bias_weight_range = initial_bias_weight_range
        self.time_lags = time_lags
        self.cost_function = cost_function
        self.learning_method = learning_method
        self.case_manager = case_manager
        self.training_cases = case_manager.get_training_cases()
        self.validation_cases = case_manager.get_validation_cases()
        self.test_cases = case_manager.get_testing_cases()
        self.softmax = softmax
        self.validation_interval = validation_interval
        self.show_interval = show_interval
        self.monitored_variables = []  # Variables to be monitored (by gann code) during a run.

        self.build_network()


    def build_network(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_dimensions[0]
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.time_lags+1, num_inputs), name='Input')
        input_variables = self.inputs
        input_size = num_inputs
        self.layers = []
        self.cells = []
        # Build all of the modules
        #layer_size = [input, h1, h2, h3, output]
        # i er layer nr i og outsize er storrelsen paa layer nr i
        for layer_index,number_of_neurons in enumerate(self.layer_dimensions[1:]):
            layer = l.Layer(self, layer_index, input_variables, input_size, number_of_neurons, self.time_lags)
            act_func = self.get_activation_function(layer_index)
            num_units = number_of_neurons
            self.layers.append(layer)

            cell = tf.contrib.rnn.BasicRNNCell(num_units, activation=act_func)
            self.cells.append(cell)

            #input_variables = layer.output_variables
            #input_size = layer.output_size

        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.cells)
        self.output_variables, self.states = tf.nn.dynamic_rnn(multi_layer_cell, self.inputs, dtype=tf.float32)

        self.targets = tf.placeholder(tf.float32, shape=(None, self.time_lags+1,layer.output_size), name='Target')

        self.configure_learning()

#not finished this method
    def configure_learning(self):
        if (self.cost_function == "cross_entropy"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self.output_variables),
                                       name="CrossEntropy")

        elif (self.cost_function == "mean_square"):
            self.loss = tf.reduce_mean(tf.square(self.targets - self.output_variables), name='MSE')
        else:
            raise ValueError('Cost function does not exist')

        # Defining the training operator
        #TODO: add more learning methods. Gradient decent is slowly learning.
        if (self.learning_method == "gradient_decent"):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.trainer = optimizer.minimize(self.loss, name='Backprop')
        else:
            raise ValueError('Learning method does not exist')

    def run(self,epochs=100,sess=None,continued=False):
        PLT.ion()
        self.training_session(epochs,sess=sess,continued=continued)
        self.test_on_trains(sess=self.current_session) #tst on trainning set
        self.testing_session(sess=self.current_session)
        self.close_current_session()
        PLT.ioff()

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        self.roundup_probes()
        session = sess if sess else flt.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session,self.case_manager.get_training_cases(),epochs,continued=continued)

    #Continued means not first session with training
    def do_training(self, sess, cases, epochs=100, continued=False):
        if not(continued): self.error_history = []
        for epoch in range(epochs):
            error = 0
            #step = self.global_training_step + epoch (har med lagring av treningsdata og gjoere og continued sessions)
            grabbed_variables = [self.loss] + self.monitored_variables
            mbs = self.minibatch_size
            ncases = len(cases)
            number_of_batches = math.ceil(ncases/mbs)
            for case_start in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                case_end = min(ncases, case_start+mbs)
                minibatch = cases[case_start:case_end]
                inputs = [case[0] for case in minibatch]
                targets = [case[1] for case in minibatch]
                feeder = {self.inputs: inputs, self.targets: targets}
                _,grabvals,_ = self.run_one_step([self.trainer], grabbed_variables, self.probes, session=sess,
                                         feed_dict=feeder, step=epoch, show_interval=self.show_interval)
                error += grabvals[0]
            self.error_history.append((epoch, error/number_of_batches))
            self.consider_validation_testing(epoch,sess)
        self.global_training_step += epochs
        flt.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
                                  title="",fig=not(continued))

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else flt.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess


    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess,cases,msg='Validation Testing')
                self.validation_history.append((epoch,error))

    def display_grabvars(self, grabbed_vals, grabbed_vars,step=1):
        names = [x.name for x in grabbed_vars];
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg)
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", "\n")
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                flt.hinton_plot(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                fig_index += 1
            else:
                print(v, "\n")


    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def get_activation_function(self, layer_index):
        act = self.activation_functions[layer_index]
        if (act == "relu"):
            return tf.nn.relu
        elif (act == "sigmoid"):
            return tf.nn.sigmoid
        elif (act == "tanh"):
            return tf.nn.tanh
        else:
            raise ValueError("Wrong activation function")

    def get_initial_weight_range(self):
        return self.initial_weight_range

    def get_initial_bias_weigh_range(self):
        return self.initial_bias_weight_range
