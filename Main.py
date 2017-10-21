import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time

#Standarized names for activation_functions:    "relu" - Rectified linear unit
#                                               "sigmoid" - Sigmoid
#                                               "tanh" - Hyperbolic tangens

#Standarized names for cost functions:          "cross_entropy" - Cross entropy
#                                               "mean_square" - Mean square error

#Standarized names for learningn method:        "gradient_decent" - Gradient decent

def Main():
    print("--- READING DATA ---")
    start_time = time.time()
    fromDate = "01.01.2005"
    number_of_trading_days = 100
    attributes_input = ["op", "cp"]
    attributes_output = ["ret"]
    one_hot_vector_interval = [-0.000, 0.000]

    selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
    selectedFTSE100 = ssr.readSelectedStocks("FTSE100.txt")
    sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, fromDate, "S&P500.txt", 7,
                                         number_of_trading_days, normalize_method="minmax", start_time=start_time)
    lftse100 = pi.InputPortolfioInformation(selectedFTSE100, attributes_output, fromDate, "LFTSE100wReturn.txt", 1,
                                            number_of_trading_days, normalize_method="minmax",
                                            one_hot_vector_interval=one_hot_vector_interval, is_output=True,
                                            start_time=start_time)
    # selectedSP500 = ssr.readSelectedStocks("TestInput.txt")
    # selectedFTSE100 = ssr.readSelectedStocks("TestOutput.txt")
    # sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, fromDate, "det-Input.txt", 7,
    #                                      number_of_trading_days, normalize_method="minmax", start_time=start_time)
    # lftse100 = pi.InputPortolfioInformation(selectedFTSE100, attributes_output, fromDate, "det-Output.txt", 1,
    #                                                 number_of_trading_days, normalize_method="minmax",
    #                                                 one_hot_vector_interval=one_hot_vector_interval, is_output=True,
    #                                                 start_time=start_time)

    time_lags = 0
    one_hot_size = 3
    #print(cases)

    learning_rate = 0.1
    minibatch_size = 10
    activation_functions = ["relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu"]
    initial_weight_range = [-1.0, 1.0]
    initial_bias_weight_range = [0.0, 0.0]
    cost_function = "cross_entropy"
    learning_method = "gradient_decent"
    validation_interval = None
    show_interval = None
    softmax=True

    number_of_networks = 2
    case_generator = cg.CaseGenerator(sp500.normalized_portfolio_data, lftse100.portfolio_data,
                                      lftse100.normalized_portfolio_data, time_lags, one_hot_vector_interval,
                                      one_hot_size)
    cases = case_generator.cases
    fraction_of_cases_for_one_network = float(1.0/float(number_of_networks))
    print("Frac: " + str(fraction_of_cases_for_one_network))
    seperator0 = 0
    accuracies = []
    accuracy_info_list = []
    over_all_returns = []
    total_accuracy_sum = 0.0
    total_testing_cases = 0.0
    for network_nr in range(0, number_of_networks):
        print ("\n--- BUILDING NEURAL NET " + str(network_nr) + "\t %s seconds ---" % (time.time() - start_time))
        separator1 = int(round(len(cases) * fraction_of_cases_for_one_network))+seperator0
        if (network_nr == number_of_networks-1):
            separator1=len(cases)
        case_manager = cm.CaseManager(cases[seperator0:separator1], time_lags, validation_fraction=0.0, test_fraction=0.10)

        input_size = len(cases[0][0][0])
        output_size = len(cases[0][1][0])
        layer_dimension = [input_size, output_size]

        neural_net = nn.NeuralNet(layer_dimension, activation_functions, learning_rate,
                                minibatch_size, initial_weight_range, initial_bias_weight_range,
                                time_lags, cost_function, learning_method, case_manager, validation_interval,
                                show_interval, softmax, start_time)
        neural_net.run(epochs=1, sess=None, continued=None)

        accuracies.append(neural_net.accuracy)
        accuracy_info_list.append(neural_net.accuracy_information)
        over_all_returns.append(neural_net.overall_return)
        total_accuracy_sum += neural_net.accuracy*neural_net.testing_size

        total_testing_cases += neural_net.testing_size

        seperator0 = separator1
    accuracy = total_accuracy_sum/total_testing_cases
    total_acc_info = generate_total_acc_info(accuracy_info_list, number_of_networks)
    over_all_return = generate_over_all_return(over_all_returns)
    print_accuracy_info(total_acc_info, accuracies, over_all_return, over_all_returns)



def printCases(cases):
    day = 1
    for case in cases:
        print("DAY " + str(day))
        print("Input: "+ str(case[0]))
        print("Output: "+ str(case[1]))
        day+=1

def print_accuracy_info(total_acc_info, accuracies, over_all_return, over_all_returns):
    print("--- Over all accuracies: " + str(accuracies) + " ---")
    print("--- Over all accuracy categorization: ")
    for true_false in total_acc_info:
        for classification in total_acc_info[true_false]:
            print("\t--- " + true_false + "-" + classification + ": " + str(total_acc_info[true_false][classification]))
    print("--- Over all returns: " + str(over_all_returns))
    print("--- Total return: " +"{0:.4f}%".format((over_all_return-1) * 100))

def generate_total_acc_info(accuracy_info_list, number_of_networks):
    total_acc_info = feed_accuracy_relevant_dictionaries()
    total_numbers = feed_accuracy_relevant_dictionaries()
    for true_false in accuracy_info_list[0]:
        for classification in accuracy_info_list[0][true_false]:
            for acc_info in accuracy_info_list:
                info = acc_info[true_false][classification]
                total_numbers[true_false][classification] += info
    for true_false in accuracy_info_list[0]:
        for classification in accuracy_info_list[0][true_false]:
            total_acc_info[true_false][classification] = (float(total_numbers[true_false][classification])/float(number_of_networks))
    return total_acc_info

def generate_over_all_return(over_all_returns):
    tot_ret = 1.0
    for ret in over_all_returns:
        tot_ret *= ret
    return tot_ret

def feed_accuracy_relevant_dictionaries():
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

Main()