import numpy as np
import copy


class CaseGenerator():

    def __init__(self, input_data, output_data, time_lags):
        self.cases = []
        days_input_data = []
        days_output_data = []
        for i in range(time_lags):
            empty_day_list_input = []
            empty_day_list_output = []
            for key in input_data:
                empty_day_list_input.extend(np.zeros(len(input_data[key][0])))
            days_input_data.append(empty_day_list_input)
            empty_day_list_output.extend(np.zeros(len(output_data[0][0])))
            days_output_data.append(empty_day_list_output)
        day_nr = 0
        number_of_days = len(output_data[0])
        print("Output len: " + str(len(output_data[0])))
        print("Input len: " + str(len(input_data[0])))
        for i in range(0, number_of_days):
            day_input_data = []
            for key in input_data:
                day_input_data.extend(input_data[key][day_nr])

            days_input_data.append(day_input_data)
            days_output_data.append(output_data[0][day_nr])
            #day_output_data = output_data[0][day_nr]
            case = [days_input_data, days_output_data]
            day_nr += 1
            self.cases.append(case)
            days_input_data = copy.deepcopy(days_input_data)
            days_output_data = copy.deepcopy(days_output_data)
            del days_input_data[0]
            del days_output_data[0]

