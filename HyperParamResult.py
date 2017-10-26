import Main as m
import StockResult as sr


class HyperParamResults():

    def __init__(self, tot_return, tot_long_return, tot_short_return, #returns on different strategies
             tot_day_std, tot_day_short_std, tot_day_long_std, #standard deviations
             aggregate_counter_table, hyper_param_dict, #dict for [pred][actual] and hyperparams
             stock_accuracies, stock_returns, stock_short_returns, stock_long_returns,
             acc_up_up, acc_up_d, acc_up_s, acc_d_up, acc_d_s, acc_d_d, acc_s_s, acc_s_d, acc_s_up,
             prec_up_up, prec_up_d, prec_up_s, prec_d_up, prec_d_s, prec_d_d, prec_s_s, prec_s_d, prec_s_up,):

        #Returns on complete portfolio
        self.tot_return = tot_return
        self.tot_long_return = tot_long_return
        self.tot_short_return = tot_short_return

        # Standard deviations per day:
        self.tot_day_std = tot_day_std
        self.tot_day_short_std = tot_day_short_std
        self.tot_day_long_std = tot_day_long_std

        #Accuracy
        self.acc_up_up = acc_up_up
        self.acc_up_d = acc_up_d
        self.acc_up_s = acc_up_s

        self.acc_d_d = acc_d_d
        self.acc_d_u = acc_d_up
        self.acc_d_s = acc_d_s

        self.acc_s_s = acc_s_s
        self.acc_s_u = acc_s_up
        self.acc_s_d = acc_s_d

        #Precision
        self.prec_up_up = prec_up_up
        self.prec_up_d = prec_up_d
        self.prec_up_s = prec_up_s

        self.prec_d_d = prec_d_d
        self.prec_d_u = prec_d_up
        self.prec_d_s = prec_d_s

        self.prec_s_s = prec_s_s
        self.prec_s_u = prec_s_up
        self.prec_s_d = prec_s_d

        #Dictionary containing count on each prediction,actual pair. Keys: "up", "down", "stay"
        self.aggregate_counter_table = aggregate_counter_table

        #Dictionary with hyperparameters. Keys:
        # "activation_functions",
        # "hidden_layer_dimension"
        # "time_lags"
        # "one_hot_vector_interval"
        # "keep_probability_dropout"
        # "from_date"
        # "number_of_trading_days"
        # "attributes_input"
        # "learning_rate"
        # "minibatch_size"
        # "number_of_stocks"
        # "epochs"

        self.hyper_param_dict = hyper_param_dict

        #Per stock results aggregated over all the runs
        self.stock_accuracies = stock_accuracies
        self.stock_returns = stock_returns
        self.stock_short_returns = stock_short_returns
        self.stock_long_returns = stock_long_returns
