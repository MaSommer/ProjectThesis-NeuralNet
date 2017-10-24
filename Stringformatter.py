class Stringformatter():

    def __init__(self):
        output_string = ""

    def generate_result_string(self, stock_result):
        stock_name = "Statoil"

        pred_total_return = 12.00           #result.get_total_return()
        pred_total_return_up = 3.50         #result.get_total_up_return()
        pred_total_return_down = -2.00      #result.get_total_return_down()

        pred_total_accuracy = 48.00         #result.get_pred_total_accuracy()
        pred_total_accuracy_up = 51.00      #result.get_pred_total_accuracy_up()
        pred_total_accuracy_down = 47.00    #result.get_pred_total_accuracy_down()

        pred_periodic_returns = [1.00,2.33,1.44,-3.00]  #result.get_periodic_returns()
        pred_periodic_accuracies = [55.00,43.00,51.00,52.33] #result.get_periodic_accuracies()

        pred_periodic_returns_up = [1.00,2.33,1.44,-3.00]     #result.get_periodic_returns_up()
        pred_periodic_accuracies_up = [55.00,43.00,51.00,52.33] # result.get_periodic_accuracies_up()

        pred_periodic_returns_down = [1.00, 2.33, 1.44, -3.00]  # result.get_periodic_returns_down()
        pred_periodic_accuracies_down = [55.00, 43.00, 51.00, 52.33]  # result.get_periodic_accuracies_down()

        pred_actual_counts = stock_result

        

