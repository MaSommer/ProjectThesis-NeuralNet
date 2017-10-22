import read_stock_data.SelectedStockReader as ssr
import read_stock_data.InputPortfolioInformation as pi
import read_stock_data.CaseGenerator as cg
import neural_net.CaseManager as cm
import neural_net.NeuralNet as nn
import time
import os
import Result as res

#Standarized names for activation_functions:    "relu" - Rectified linear unit
#                                               "sigmoid" - Sigmoid
#                                               "tanh" - Hyperbolic tangens

#Standarized names for cost functions:          "cross_entropy" - Cross entropy
#                                               "mean_square" - Mean square error

#Standarized names for learningn method:        "gradient_decent" - Gradient decent

def run_over_night():
    f = open("res.txt", "w");
    selected = generate_selected_list()
    number_of_stocks_to_test = 2
    selectedSP500 = ssr.readSelectedStocks("S&P500.txt")

    for stock_nr in range(0, number_of_stocks_to_test):
        selected[stock_nr] = 1
        selectedFTSE100 = selected
        result = Main(selectedSP500, selectedFTSE100)
        write_result_to_file(result, stock_nr)

        selected[stock_nr] = 0


def Main(selectedSP500, selectedFTSE100):
    print("--- READING DATA ---")
    start_time = time.time()
    fromDate = "01.01.2001"
    number_of_trading_days = 50
    attributes_input = ["op", "cp"]
    attributes_output = ["ret"]
    one_hot_vector_interval = [-0.003, 0.003]

    #selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
    #selectedFTSE100 = ssr.readSelectedStocks("FTSE100.txt")
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

    time_lags = 3
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
    result = res.Result(start_time)
    for network_nr in range(0, number_of_networks):
        print ("\n--- BUILDING NEURAL NET " + str(network_nr) + "\t %s seconds ---" % (time.time() - start_time))
        separator1 = int(round(len(cases) * fraction_of_cases_for_one_network))+seperator0
        if (network_nr == number_of_networks-1):
            separator1=len(cases)
        case_manager = cm.CaseManager(cases[seperator0:separator1], time_lags, validation_fraction=0.0, test_fraction=0.10)

        input_size = len(cases[0][0][0])
        output_size = len(cases[0][1][0])
        layer_dimension = [input_size, 100, 60, 20, output_size]

        neural_net = nn.NeuralNet(layer_dimension, activation_functions, learning_rate,
                                minibatch_size, initial_weight_range, initial_bias_weight_range,
                                time_lags, cost_function, learning_method, case_manager, validation_interval,
                                show_interval, softmax, start_time)
        neural_net.run(epochs=40, sess=None, continued=None)

        result.add_to_result(neural_net)

        seperator0 = separator1

    result.generate_final_result_info(number_of_networks)
    result = result.genereate_result_string()
    return result
    #result.print_accuracy_info()


def generate_selected_list():
    selected = []
    for i in range(0, 300):
        selected.append(0)
    return selected


def write_result_to_file(result, stock):
    f = open("res.txt", "a");
    f.write("REUSLT FOR " + str(stock) + "\n" + str(result) + "\n\n")  # python will convert \n to os.linesep
    f.close()

def printCases(cases):
    day = 1
    for case in cases:
        print("DAY " + str(day))
        print("Input: "+ str(case[0]))
        print("Output: "+ str(case[1]))
        day+=1

run_over_night()