import SelectedStockReader as ssr
import PortfolioInformation as pi


#Standarized names for activation_functions:    "relu" - Rectified linear unit
#                                               "sigmoid" - Sigmoid
#                                               "tanh" - Hyperbolic tangens

#Standarized names for cost functions:          "cross_entropy" - Cross entropy
#                                               "mean_square" - Mean square error

def Main():
    fromDate = "10.08.2017"
    toDate = "11.08.2017"
    attributes = ["op", "cp", "tv"]
    selectedSP500 = ssr.readSelectedStocks("S&P500v2.txt")
    sp500 = pi.PortolfioInformation(selectedSP500, attributes, fromDate, toDate, "S&P500.txt")
    print(sp500.attributeData)

#TODO: make method for reading target file
#TODO: make method for generating cases

Main()