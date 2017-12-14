import tensorflow as tf
import numpy as np
import neural_net.CaseManager as cm
import neural_net.Layer as l
import FlowTools as flt
import matplotlib.pyplot as PLT
import math
import time
import neural_net.NeuralNetResults as res
import copy




class NeuralNet():


#layerDimensions = [input size, hidden layer1 size, hidden layer 2 size, ..., output layer size]
#activationFunctions = [act function to h1, act function to h2, ...., act function to output] write them as string
#initial_weight_range = [start_range, stop_range], ex [-1.0, 1.0]
#cases = [[input1, input2, input3, ...], [target1, target2, ...]]
#time_lags is number of steps in previous states in the reccurent network
#learning_method as a string, example "gradient_decent"
#number_of_target_possibilities is [down, same, up] or [-inf to -1, -1 to 1, 1 to 5, 5 to inf]
#validation_interval is how often

    def __init__(self, network_nr, layer_dimensions, activation_functions, learning_rate, minibatch_size, time_lags, cost_function,
                 learning_method, case_manager, keep_probability_for_dropout, validation_interval=None, show_interval=None,
                 softmax=True, start_time=time.time(), rank = 0):
        self.network_nr = network_nr
        self.keep_probability_for_dropout = keep_probability_for_dropout
        self.layer_dimensions = layer_dimensions
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.activation_functions = activation_functions
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

        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.validation_history = []

        self.start_time = start_time
        self.rank = rank
        self.probes = None
        self.training_acc= 0
        self.test_acc = 0

        self.build_network()


    def build_network(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_dimensions[0]
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.time_lags+1, num_inputs), name='Input')
        input_variables = self.inputs
        input_size = num_inputs
        self.layers = []
        self.cells = []
        self.drop_cells = []
        # Build all of the modules
        #layer_size = [input, h1, h2, h3, output]
        # i er layer nr i og outsize er storrelsen paa layer nr i
        for layer_index,number_of_neurons in enumerate(self.layer_dimensions[1:]):
            layer = l.Layer(self, layer_index, input_variables, input_size, number_of_neurons, self.time_lags)

            act_func = self.get_activation_function(layer_index)
            num_units = number_of_neurons
            self.layers.append(layer)

            cell = tf.contrib.rnn.BasicRNNCell(num_units, activation=act_func)
            if (layer_index == 0): #indicates that it is the first hidden layer
                drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_probability_for_dropout[0])
            else:
                drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_probability_for_dropout[1])

            self.cells.append(cell)
            self.drop_cells.append(drop_cell)

            #input_variables = layer.output_variables
            #input_size = layer.output_size

        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.drop_cells)
        self.output_variables, self.states = tf.nn.dynamic_rnn(multi_layer_cell, self.inputs, dtype=tf.float32)
        self.targets = tf.placeholder(tf.float32, shape=(None, self.time_lags+1, layer.output_size), name='Target')

        self.configure_learning()

#not finished this method
    def configure_learning(self):
        trans_output = (tf.transpose(self.output_variables, [1, 0, 2]))
        trans_target = (tf.transpose(self.targets, [1, 0, 2]))
        if (self.cost_function == "cross_entropy"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=trans_target[-1], logits=trans_output[-1]),
                                       name="CrossEntropy")

        elif (self.cost_function == "mean_square"):
            self.loss = tf.reduce_mean(tf.square(trans_target[-1] - trans_output[-1]), name='MSE')
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
        #PLT.ion()
        #print ("\tProcessor #" + str(self.rank) + "\tTraining net #" + str(self.network_nr) + "\t %s seconds ---" % (time.time() - self.start_time))
        self.training_session(epochs,sess=sess,continued=continued)
        #print ("\t\t Processor #" + str(self.rank) + "\tTesting net #" + str(self.network_nr) + "\t %s seconds ---" % (time.time() - self.start_time))

        self.test_on_training_set(sess=self.current_session) #tst on trainning set
        self.testing_session(sess=self.current_session)
        #self.close_current_session()
        # PLT.ioff()

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        #self.roundup_probes()
        session = sess if sess else flt.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session,self.case_manager.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess):
        cases = self.case_manager.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing', is_training=False)

    #test on training set
    def test_on_training_set(self, sess):
        self.do_testing(sess, self.case_manager.get_training_cases(), msg='Total Training', is_training=True)

    #Continued means not first session with training
    def do_training(self, sess, cases, epochs=100, continued=False):
        if not(continued): self.error_history = []
        for epoch in range(epochs):
            error = 0
            step = self.global_training_step + epoch
            #step = self.global_training_step + epoch (har med lagring av treningsdata og gjoere og continued sessions)
            grabbed_variables = [self.loss] + self.monitored_variables
            mbs = self.minibatch_size
            ncases = len(cases)
            number_of_batches = math.ceil(ncases/mbs)
            for case_start in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                case_end = min(ncases, case_start+mbs)
                minibatch = cases[case_start:case_end]
                inputs = ([case[0] for case in minibatch])
                targets = ([case[1] for case in minibatch])
                feeder = {self.inputs: inputs, self.targets: targets}
                _,grabvals,_ = self.run_one_step([self.trainer], grabbed_variables, self.probes, session=sess,
                                         feed_dict=feeder, step=step, show_interval=self.show_interval)
                error += grabvals[0]
                #accuracy = trainer_and_accuracy[1]
            self.error_history.append((epoch, error/number_of_batches))
            self.consider_validation_testing(epoch,sess)
            #if (epoch == epochs):
                #print("\t\tProcessor #" + str(self.rank) + "finished training net #" + str(self.network_nr) + " after \t %s seconds ---" % (time.time() - self.start_time))
        self.global_training_step += epochs
        loss_train = []
        epochs_train = []
        for i in range(len(self.error_history)):
            loss_train.append(self.error_history[i][1])
            epochs_train.append(self.error_history[i][0])
        loss_validation = []
        epochs_validation = []
        for i in range(len(self.validation_history)):
            loss_validation.append(self.validation_history[i][1])
            epochs_validation.append(self.validation_history[i][0])
        PLT.plot(epochs_train, loss_train)
        PLT.plot(epochs_validation, loss_validation)
        PLT.show()
        #flt.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
        #                          title="",fig=not(continued))
        k = input("string:")

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else flt.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            #sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def do_testing(self,sess,cases,msg='Testing', is_training = False):
        trans_output = (tf.transpose(self.output_variables, [1, 0, 2]))
        trans_target = (tf.transpose(self.targets, [1, 0, 2]))
        correct_pred = tf.equal(tf.argmax(trans_output[-1], 1), tf.argmax(trans_target[-1], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        trans_output_print = tf.argmax((trans_output)[-1], 1)
        trans_target_print = tf.argmax((trans_target)[-1], 1)


        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        returns = [c[2] for c in cases]

        feeder = {self.inputs: inputs, self.targets: targets}

        acc_and_correct_pred, grabvals, _ = self.run_one_step(
            [accuracy, correct_pred, trans_output_print, trans_target_print], self.loss, self.probes,
            session=sess, feed_dict=feeder, show_interval=None)
        #print('%s Set Accuracy = %f ' % (msg, acc_and_correct_pred[0]) + " on test size: " + str(len(cases)))
        if (is_training):
            self.training_acc = float(str(acc_and_correct_pred[0]))
        else:
            self.accuracy = float(str(acc_and_correct_pred[0]))
            self.testing_size = float((len(cases)))

            self.results = res.NeuralNetResults(self, acc_and_correct_pred[2], acc_and_correct_pred[3], returns)
        return accuracy, grabvals

    def convert_tensor_list_to_list(self, tensor_info):
        tensor_string = str(tensor_info)
        tensor_list_of_string = tensor_string.replace("[", "").replace("]", "").split(" ")
        tensor_list_of_int = []
        for i in range (0,len(tensor_list_of_string)):
            tensor_list_of_int.append(int(tensor_list_of_string[i]))
        return tensor_list_of_int

    def generate_accuracy_information_and_overall_return(self, predication_list, target_list, returns):
        self.overall_return = 1.0
        counter_dict = self.feed_accuracy_relevant_dictionaries()
        accuracy_info = self.feed_accuracy_relevant_dictionaries()
        true_false_counter = {}
        true_false_counter["true"] = 0
        true_false_counter["false"] = 0
        correct_pred = True
        self.number_of_correct_predication_beginning_streak = 0

        for i in range(0, len(predication_list)):
            pred = predication_list[i]
            target = target_list[i]
            return_that_day = returns[i]
            if (pred != target):
                correct_pred = False
                self.update_accuracy_counter(counter_dict, "false", true_false_counter, pred)
                self.update_return(return_that_day, pred, "false", target)
            else:
                if (correct_pred):
                    self.number_of_correct_predication_beginning_streak+=1
                self.update_accuracy_counter(counter_dict, "true", true_false_counter, pred)
                self.update_return(return_that_day, pred, "true", target)
        for true_false in counter_dict:
            for classification in counter_dict[true_false]:
                current_count = counter_dict[true_false][classification]
                if (true_false_counter[true_false] == 0):
                    accuracy_info[true_false][classification] = 0
                else:
                    accuracy_info[true_false][classification] = float(float(current_count)/float(true_false_counter[true_false]))
        return accuracy_info

    def update_accuracy_counter(self, counter_dict, true_false, true_false_counter, pred):
        true_false_counter[true_false] += 1
        if (pred == 0):
            current = counter_dict[true_false]["down"]
            counter_dict[true_false]["down"] = current + 1
        elif (pred == 1):
            current = counter_dict[true_false]["stay"]
            counter_dict[true_false]["stay"] = current + 1
        else:
            current = counter_dict[true_false]["up"]
            counter_dict[true_false]["up"] = current + 1

#updates the over all return, assume no transaction costs
    def update_return(self, return_that_day, pred, true_false, target):
        if (true_false == "true"):
            if (pred == 0 and target == 0):
                self.overall_return *= (-return_that_day+1)
            elif(pred == 2 and target == 2):
                self.overall_return *= (return_that_day+1)
        else:
            if (pred == 0 and target == 2):
                self.overall_return *= (1-return_that_day)
            elif(pred == 2 and target == 0):
                self.overall_return * (1+return_that_day)
            elif(pred == 0 and target == 1):
                self.overall_return *= (1 - return_that_day)
            elif(pred == 2 and target == 1):
                self.overall_return *= (1 + return_that_day)


    def feed_accuracy_relevant_dictionaries(self):
        dictionary = {}
        dictionary["false"] = {}
        # the list is [number of false, number of false because up]
        dictionary["false"]["up"] = 0
        dictionary["false"]["stay"] = 0
        dictionary["false"]["down"] = 0
        dictionary["true"] = {}
        dictionary["true"]["up"] = 0
        dictionary["true"]["stay"] = 0
        dictionary["true"]["down"] = 0
        return dictionary


    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.case_manager.get_validation_cases()
            if len(cases) > 0:
                error, grabvals = self.do_testing(sess,cases,msg='Validation Testing')
                self.validation_history.append((epoch,grabvals))

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


    def runmore(self,epochs=100):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True)

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for layer in self.layers:
            vars = [layer.getvar('wgt'), layer.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = flt.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

#TODO: Fix the save session_params method. Currently an error is occurring.
    def close_current_session(self):
        #self.save_session_params(sess=self.current_session)
        flt.close_session(self.current_session, view=True)

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_monitored_variables(self, module_index, type='wgt'):
        self.monitored_variables.append(self.layers[module_index].getvar(type))
        #self.grabvar_figures.append(PLT.figure())

    def get_activation_function(self, layer_index):
        act = self.activation_functions[layer_index]
        if (act == "relu"):
            return tf.nn.relu
        elif (act == "sigmoid"):
            return tf.nn.sigmoid
        elif (act == "tanh"):
            return tf.nn.tanh
        elif (act == "lin"):
            return None
        else:
            raise ValueError("Wrong activation function")

