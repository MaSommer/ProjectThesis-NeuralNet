import numpy as np
import copy


class CaseGenerator():

    def __init__(self, input_data, output_data, output_normalized_data, time_lags, one_hot_vector_interval, one_hot_size):
        self.cases = []
        self.one_hot_vector_interval = one_hot_vector_interval
        self.one_hot_size = one_hot_size
        self.days_input_data = []
        self.days_output_data = []
        self.addZeros(input_data, output_data, time_lags)
        # print("Output len: " + str(len(output_data[0])))
        # print("Input len: " + str(len(input_data[0])))
        self.add_data(input_data, output_data, output_normalized_data, time_lags)


    #Adding zeros for the first time lags because we do not have data for day t-1 in the beginning
    def addZeros(self, input_data, output_data, time_lags):
        for i in range(time_lags):
            empty_day_list_input = []
            empty_day_list_output = []
            for key in input_data:
                empty_day_list_input.extend(np.zeros(len(input_data[key][0])))
            for day in range(0, time_lags):
                empty_day_list_input.append(0.0)
            self.days_input_data.append(empty_day_list_input)
            empty_day_list_output.extend(np.zeros(self.one_hot_size))
            self.days_output_data.append(empty_day_list_output)

    def add_data(self, input_data, output_data, output_normalized_data, time_lags):
        number_of_days = len(output_data[0])

        for day_nr in range(0, number_of_days):
            day_input_data = []
            for key in input_data:
                day_input_data.extend(input_data[key][day_nr])
            #adds the previous day return to the case
            for key in output_normalized_data:
                for day in range(1, time_lags + 1):
                    if (day_nr - day < 0):
                        day_input_data.append(0.0)
                    else:
                        day_input_data.extend(output_normalized_data[key][day_nr - day])

            self.days_input_data.append(day_input_data)
            one_hot_vector = self.generate_one_hot_vector(output_data[0][day_nr][0])
            self.days_output_data.append(one_hot_vector)
            # day_output_data = output_data[0][day_nr]
            case = [self.days_input_data, self.days_output_data, output_data[0][day_nr][0]]

            day_nr += 1
            self.cases.append(case)
            self.days_input_data = copy.deepcopy(self.days_input_data)
            self.days_output_data = copy.deepcopy(self.days_output_data)
            del self.days_input_data[0]
            del self.days_output_data[0]

    def generate_one_hot_vector(self, data):
        if (data < self.one_hot_vector_interval[0]):
            return [1, 0, 0]
        elif (data > self.one_hot_vector_interval[1]):
            return [0, 0, 1]
        else:
            return [0, 1, 0]