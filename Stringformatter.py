class Stringformatter():

    def __init__(self):
        self.output_string = ""

    def generate_stock_result_string(self, stock_result):
        stock_name = str(stock_result.get_stock_name())
        self.output_string += stock_name + " | Trading days: " + str(stock_result.get_number_of_days()) + "\n"

        #adding line:
        for i in range(110):
            self.output_string+="-"

        self.output_string +="\n"

        pred_total_return = "%.2f" % ((stock_result.get_over_all_return()-1)*100)
        pred_total_return_up = "%.2f" % ((stock_result.get_tot_up_return() -1)*100)         #result.get_total_up_return()
        pred_total_return_down = "%.2f" % ((stock_result.get_tot_down_return()-1)*100)

        self.output_string += "Return:\t\t" + str(pred_total_return) + "%\t\t" + "Return up:\t\t" + str(pred_total_return_up) + "%\t\t" + "Return down:" + "\t" + str(pred_total_return_down) + "%\n"
        self.output_string +="\n"
        pred_total_accuracy = "%.2f" % (stock_result.get_total_pred_accuracy()*100)
        #pred_total_accuracy_up = "%.2f" % 51.00444      #result.get_pred_total_accuracy_up()
        #pred_total_accuracy_down = "%.2f" % 47.0032535    #result.get_pred_total_accuracy_down()

        self.output_string += "Accuracy:\t" + str(pred_total_accuracy) + "%\t"# + "Accuracy up:\t" + str(pred_total_accuracy_up) + "%\t" + "Accuracy down:" + "\t" + str(pred_total_accuracy_down) + "%\n"
        self.output_string += "\n"
        # adding line:
        for i in range(110):
            self.output_string += "-"

        self.output_string += "\n"

        pred_periodic_accuracies = stock_result.get_pred_accuracies()
        pred_periodic_returns = stock_result.get_periodic_returns()

        for i in range(len(pred_periodic_accuracies)):
            a = "%.2f" % float(pred_periodic_accuracies[i])
            pred_periodic_accuracies[i] = a
            b= "%.2f" % float(pred_periodic_returns[i])
            pred_periodic_returns[i] = b

        self.output_string += "Returns per period:\t\t\t" + str(pred_periodic_returns) + "\n"
        self.output_string += "Accuracies per period:\t\t" + str(pred_periodic_accuracies) + "\n"

        self.output_string += "\n"

        pred_periodic_returns_up = stock_result.get_periodic_up_returns()
        # pred_periodic_accuracies_up = [55.00,43.00,51.00,52.33] # result.get_periodic_accuracies_up()
        #
        for i in range(len(pred_periodic_returns_up)):
             a = "%.2f" % float(pred_periodic_returns_up[i])
             pred_periodic_returns_up[i] = a
        #     b = "%.2f" % pred_periodic_accuracies_up[i]
        #     pred_periodic_accuracies_up[i] = b
        #
        self.output_string += "Returns up per period:\t\t" + str(pred_periodic_returns_up) + "\n"
        # self.output_string += "Accuracies up per period:\t" + str(pred_periodic_accuracies_up) + "\n\n"
        #
        #
        pred_periodic_returns_down = stock_result.get_periodic_down_returns()
        # pred_periodic_accuracies_down = [55.00, 43.00, 51.00, 52.33]  # result.get_periodic_accuracies_down()
        #
        for i in range(len(pred_periodic_returns_down)):
             a = "%.2f" % float(pred_periodic_returns_down[i])
             pred_periodic_returns_down[i] = a
        #     b = "%.2f" % pred_periodic_accuracies_down[i]
        #     pred_periodic_accuracies_down[i] = b
        #
        self.output_string += "Returns d per period:\t\t" + str(pred_periodic_returns_down) + "\n"
        # self.output_string += "Accuracies d per period:\t" + str(pred_periodic_accuracies_down) + "\n\n"

        estimated_and_actual_totals_count = self.generate_total_count_numbers(stock_result)

        counter_dictionaries = stock_result.get_counter_dictionaries()
        u_u = 0
        u_s = 0
        u_d = 0
        s_u = 0
        s_s = 0
        s_d = 0
        d_u = 0
        d_s = 0
        d_d = 0
        for i in range(len(counter_dictionaries)):
            u_u += counter_dictionaries[i]["up"]["up"]
            u_s += counter_dictionaries[i]["up"]["stay"]
            u_d += counter_dictionaries[i]["up"]["down"]
            s_u += counter_dictionaries[i]["stay"]["up"]
            s_s += counter_dictionaries[i]["stay"]["stay"]
            s_d += counter_dictionaries[i]["stay"]["down"]
            d_u += counter_dictionaries[i]["down"]["up"]
            d_s += counter_dictionaries[i]["down"]["stay"]
            d_d += counter_dictionaries[i]["down"]["down"]

        self.output_string += "\t\t\t\t\tActual values"
        self.output_string += "\n"
        self.output_string += "\t\t\t\t\t  |\tU" + "\t" + "S" + "\t" + "D" + "\t" + "|" + "\n"
        for i in range(19):
            self.output_string += " "
        for i in range (20):
            self.output_string += "-"
        self.output_string += "\n"
        self.output_string += "\t\t\t\t\tU |" + "\t" + str(int(u_u)) + "\t" + str(int(u_s)) + "\t" + str(int(u_d)) + "\t" + "| " + str(int(estimated_and_actual_totals_count[3])) + "\n"
        self.output_string += "Predicted values\tS |\t" + str(int(s_u)) + "\t" + str(int(s_s)) + "\t" + str(int(s_d)) + "\t" + "| " + str(int(estimated_and_actual_totals_count[4])) +"\n"
        self.output_string += "\t\t\t\t\tD |\t" + str(int(d_u)) + "\t" + str(int(d_s)) + "\t" + str(int(d_d)) + "\t" + "| " + str(int(estimated_and_actual_totals_count[5]))+"\n"

        for i in range(19):
            self.output_string += " "
        for i in range (20):
            self.output_string += "-"
        self.output_string += "\n"
        self.output_string += "\t\t\t\t\t  |\t" + str(int(estimated_and_actual_totals_count[0])) + "\t" +  str(int(estimated_and_actual_totals_count[1])) + "\t" + str(int(estimated_and_actual_totals_count[2])) + "\t|"

        self.output_string += "\n\n\n"
        return self.output_string



    def generate_total_count_numbers(self, result):
        actual_map_list = result.get_actual_map_list()
        estimated_map_list = result.get_estimated_map_list()
        actual_up = 0
        actual_down = 0
        actual_stay = 0

        estimated_stay = 0
        estimated_up = 0
        estimated_down = 0

        for dictionary in actual_map_list:
            actual_up += dictionary["up"]
            actual_down += dictionary["down"]
            actual_stay += dictionary["stay"]

        for dictionary in estimated_map_list:
            estimated_up += dictionary["up"]
            estimated_down += dictionary["down"]
            estimated_stay += dictionary["stay"]
        temp = []

        temp.append(actual_up)
        temp.append(actual_stay)
        temp.append(actual_down)

        temp.append(estimated_up)
        temp.append(estimated_stay)
        temp.append(estimated_down)

        return temp
