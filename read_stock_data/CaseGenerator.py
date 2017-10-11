


class CaseGenerator():

    def __init__(self, input_data, output_data):
        self.cases = []
        day_nr = 0
        number_of_days = len(output_data[0])
        for i in range(0, number_of_days):
            day_input_data = []
            for key in input_data:
                day_input_data.extend(input_data[key][day_nr])

            day_output_data = output_data[0][day_nr]
            case = [day_input_data, day_output_data]
            day_nr += 1
            self.cases.append(case)
