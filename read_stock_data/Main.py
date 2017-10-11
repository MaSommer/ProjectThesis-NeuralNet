import SelectedStockReader as ssr
import InputPortfolioInformation as pi
import CaseGenerator as cg

#Standarized names for activation_functions:    "relu" - Rectified linear unit
#                                               "sigmoid" - Sigmoid
#                                               "tanh" - Hyperbolic tangens

#Standarized names for cost functions:          "cross_entropy" - Cross entropy
#                                               "mean_square" - Mean square error

#Standarized names for learningn method:        "gradient_decent" - Gradient decent

def Main():
    fromDate = "10.08.2017"
    toDate = "11.08.2017"
    attributes_input = ["op", "cp", "tv"]
    attributes_output = ["ret"]
    one_hot_vector_interval = [-0.005, 0.005]
    selectedSP500, number_of_selectedSP = ssr.readSelectedStocks("S&P500v2.txt")
    selectedFTSE100, number_of_selectedFTSE = ssr.readSelectedStocks("FTSE100.txt")
    sp500 = pi.InputPortolfioInformation(selectedSP500, attributes_input, fromDate, toDate, "S&P500.txt", 7)
    lftse100 = pi.InputPortolfioInformation(selectedFTSE100, attributes_output, fromDate, toDate, "outputReturnsLFTSE100.txt", 1, one_hot_vector_interval, is_output=True)

    print("SP")
    print(sp500.portfolio_data)

    print("LONDON")
    print(lftse100.portfolio_data)

    case_generator = cg.CaseGenerator(sp500.portfolio_data, lftse100.portfolio_data)

    print(case_generator.cases)
    # number_of_outcomes_for_output_stock = 3
    # input_size = number_of_selectedSP*len(attributes_input)
    # output_size = number_of_outcomes_for_output_stock*number_of_selectedFTSE
    # layer_dimension = [input_size, 800, 100, output_size]
    # learning_rate = 0.99
    # minibatch_size = 10
    # activation_functions = ["relu", "relu", "relu"]
    # initial_weight_range = [-1.0, 1.0]
    # initial_bias_weight_range = [0.0, 0.0]
    # time_lags = 5
    # cost_function = "cross_entropy"
    # learning_method = "gradient_decent"



#TODO: make method for reading target file
#TODO: make method for generating cases

Main()