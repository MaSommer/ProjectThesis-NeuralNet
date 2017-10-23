import copy

class Results():

    def __init__(self, neural_net, predication_tensor, target_tensor, actual_returns):
        self.neural_net = neural_net
        self.predication_list = self.convert_tensor_list_to_list(predication_tensor)
        self.target_list = self.convert_tensor_list_to_list(target_tensor)

        self.day_returns = []
        # print("Prediction_list and target_list: ")
        # print(self.predication_list)
        # print(self.target_list)

        self.actual_returns = actual_returns

        #maps accuracy on number of times predicted true because up, true because stay... and false because up....
        self.accuracy_information, self.precision_information = self.generate_accuracy_information_and_overall_return()

    def convert_tensor_list_to_list(self, tensor_info):
        tensor_string = str(tensor_info)
        tensor_list_of_string = tensor_string.replace("[", "").replace("]", "").split(" ")
        tensor_list_of_int = []
        for i in range(0, len(tensor_list_of_string)):
            tensor_list_of_int.append(int(tensor_list_of_string[i]))
        return tensor_list_of_int

    def generate_accuracy_information_and_overall_return(self):
        self.overall_return = 1.0
        self.initialize_estimated__and_actual_map()
        counter_dict = self.feed_accuracy_relevant_dictionaries()
        accuracy_info = self.feed_accuracy_relevant_dictionaries()
        precision_info = self.feed_accuracy_relevant_dictionaries()

        correct_pred = True
        self.number_of_correct_predication_beginning_streak = 0

        for i in range(0, len(self.predication_list)):
            pred = self.predication_list[i]
            target = self.target_list[i]
            return_that_day = self.actual_returns[i]
            self.update_estimated_or_actual_map(self.estimated_map, float(pred))
            self.update_estimated_or_actual_map(self.actual_map, float(target))
            self.update_accuracy_counter(counter_dict, pred, target)
            if (pred != target):
                correct_pred = False
                self.update_return(return_that_day, pred, "false", target)
            else:
                if (correct_pred):
                    self.number_of_correct_predication_beginning_streak += 1
                self.update_return(return_that_day, pred, "true", target)
        for predicted in counter_dict:
            for target in counter_dict[predicted]:
                self.add_to_accuracy_or_precision_info(accuracy_info, predicted, target, counter_dict, self.actual_map, target)
                self.add_to_accuracy_or_precision_info(precision_info, predicted, target, counter_dict, self.estimated_map, predicted)
        return accuracy_info, precision_info

    def add_to_accuracy_or_precision_info(self, info, prediction, target, counter_dict, divider, key):
        current_count = counter_dict[prediction][target]
        if (divider[key] == 0):
            info[prediction][target] = 0
        else:
            info[prediction][target] = float(
                float(current_count) / float(divider[key]))

    def update_estimated_or_actual_map(self, map, value):
        if (value == 0):
            map["down"] += 1
        elif(value == 1):
            map["stay"] += 1
        elif(value == 2):
            map["up"] += 1
        else:
            raise ValueError("Value is not within the classification")


    def update_accuracy_counter(self, counter_dict, pred, target):
        pred_string = self.convert_number_to_string_for_prediction(pred)
        target_string = self.convert_number_to_string_for_prediction(target)
        counter_dict[pred_string][target_string]+=1


    # updates the over all return, assume no transaction costs
    def update_return(self, return_that_day, pred, true_false, target):
        if (true_false == "true"):
            if (pred == 0 and target == 0):
                self.overall_return *= (-return_that_day + 1)
            elif (pred == 2 and target == 2):
                self.overall_return *= (return_that_day + 1)
        else:
            if (pred == 0 and target == 2):
                self.overall_return *= (1 - return_that_day)
            elif (pred == 2 and target == 0):
                self.overall_return *= (1 + return_that_day)
            elif (pred == 0 and target == 1):
                self.overall_return *= (1 - return_that_day)
            elif (pred == 2 and target == 1):
                self.overall_return *= (1 + return_that_day)
        day_return = copy.deepcopy(self.overall_return)
        self.day_returns.append(day_return)

    def feed_accuracy_relevant_dictionaries(self):
        dictionary = {}
        dictionary["up"] = {}
        # the list is [pred, target] counts how many predicted up and also target up
        dictionary["up"]["up"] = 0
        dictionary["up"]["stay"] = 0
        dictionary["up"]["down"] = 0
        dictionary["stay"] = {}
        dictionary["stay"]["up"] = 0
        dictionary["stay"]["stay"] = 0
        dictionary["stay"]["down"] = 0
        dictionary["down"] = {}
        dictionary["down"]["up"] = 0
        dictionary["down"]["stay"] = 0
        dictionary["down"]["down"] = 0
        return dictionary

    def convert_number_to_string_for_prediction(self, number):
        if (number == 0):
            return "down"
        elif(number == 1):
            return "stay"
        elif (number == 2):
            return "up"

    def initialize_estimated__and_actual_map(self):
        self.estimated_map = {}
        self.estimated_map["up"] = 0
        self.estimated_map["stay"] = 0
        self.estimated_map["down"] = 0
        self.actual_map = {}
        self.actual_map["up"] = 0
        self.actual_map["stay"] = 0
        self.actual_map["down"] = 0

