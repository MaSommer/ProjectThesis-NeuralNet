#import Main as m
import StockResult as sr


class HyperParamResult():

    def __init__(self, hyper_param_dict, per_stock_dict, result_dict):

        #dictionary with all interesting returns, stds and accuracies. Single number for every key.
        self.result_dict = result_dict

        #Dictionary with hyperparameters. Keys:

        self.hyper_param_dict = hyper_param_dict

        #Per stock results aggregated over all the runs
        self.per_stock_dict = per_stock_dict
