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
    fromDate = "01.01.2011"
    number_of_trading_days = 500
    attributes_input = ["op", "cp", "tv", "hp", "lp"]
    attributes_output = ["ret"]
    one_hot_vector_interval = [-0.0000, 0.0000]
    #selectedSP500 = ssr.readSelectedStocks("TestInput.txt")
    #selectedFTSE100 = ssr.readSelectedStocks("TestOutput.txt")
    #sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, fromDate, "testInput.txt", 7, number_of_trading_days, start_time=start_time)
    #lftse100 = pi.InputPortolfioInformation(selectedFTSE100, attributes_output, fromDate, "testOutput.txt", 1, number_of_trading_days, one_hot_vector_interval, is_output=True, start_time=start_time)

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

    time_lags = 2
    one_hot_size = 3
    case_generator = cg.CaseGenerator(sp500.normalized_portfolio_data, lftse100.portfolio_data, lftse100.normalized_portfolio_data,time_lags, one_hot_vector_interval, one_hot_size)
    cases = case_generator.cases

    case_manager = cm.CaseManager(cases, time_lags, validation_fraction=0.0, test_fraction=0.1)
    print("CASES:")
    #printCases(case_manager.cases)
    #print(cases)

    input_size = len(cases[0][0][0])
    output_size = len(cases[0][1][0])
    layer_dimension = [input_size, 800, 100, 20, output_size]
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

    print ("\n--- BUILDING NEURAL NET \t %s seconds ---" % (time.time() - start_time))
    neural_net = nn.NeuralNet(layer_dimension, activation_functions, learning_rate,
                              minibatch_size, initial_weight_range, initial_bias_weight_range,
                              time_lags, cost_function, learning_method, case_manager, validation_interval,
                              show_interval, softmax, start_time)
    neural_net.run(epochs=30, sess=None, continued=None)


def printCases(cases):
    day = 1
    for case in cases:
        print("DAY " + str(day))
        print("Input: "+ str(case[0]))
        print("Output: "+ str(case[1]))
        day+=1


Main()