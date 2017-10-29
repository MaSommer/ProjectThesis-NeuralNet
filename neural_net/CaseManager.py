import numpy as np
import copy as copy


class CaseManager():

    def __init__(self, cases, time_lag, validation_fraction=0.1, test_fraction=0.1, one_hot_vector_interval = [-0.000,0.000]):
        self.cases = cases[time_lag:len(cases)]
        self.time_lag = time_lag
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.training_fraction = 1 - (validation_fraction+test_fraction)
        self.one_hot_interval = one_hot_vector_interval
        self.organize_cases()
        self.replace_one_hot_with_soft_labels_linear(0.1)

#this method defines what is training_cases, test_cases and validation_cases
    def organize_cases(self):
        separator1 = int(round(len(self.cases) * self.training_fraction))
        separator2 = int(separator1 + round(len(self.cases) * self.validation_fraction))

        #skips the first cases due to time lag
        self.training_cases = self.cases[self.time_lag:separator1]
        self.validation_cases = self.cases[separator1:separator2]
        self.testing_cases = self.cases[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases


    def replace_one_hot_with_soft_labels_linear(self, x_percent):

        returns_per_interval = self.generate_categorized_returns_lists()
        x_percent_avgs = self.find_top_x_avgs(returns_per_interval, x_percent) #0.1 to get 10% of extremes up and down
        for case in self.training_cases:
            for i in range(0, len(case[1])):                #for ever one_hot_vector in case due to timelags
                case_ret = case[3][i]     #case[3] has the corresponding returns for the one_hot_vectors
                output_val = case[1][i]
                if(self.is_one_hot(output_val)):

                    if(case_ret > self.one_hot_interval[1]):
                        soft_l = self.generate_up_soft_label_linear(x_percent_avgs, case_ret)
                        case[1][i] = soft_l

                    elif(case_ret < self.one_hot_interval[0]):
                        soft_l = self.generate_down_soft_label_linear(x_percent_avgs, case_ret)
                        case[1][i] = soft_l
                    elif(case_ret < self.one_hot_interval[1] and case_ret > self.one_hot_interval[0]):

                        case[1][i] = self.generate_stay_soft_label_linear(case_ret)



    def generate_categorized_returns_lists(self):           # returns a list of list. list[0]=down returns, list[1] = stay, list[2] = up
        interval_up = []
        interval_down = []
        interval_stay = []

        for case in self.training_cases:  # Generates list of up, down and stay returns
            case_ret = case[2]
            if (case_ret > self.one_hot_interval[1]):  # Add for return that is going up by the range definition
                interval_up.append(case_ret)

            if (case_ret < self.one_hot_interval[0]):  # add return going down
                interval_down.append(case_ret)

            else:  # add return that stays, according to definition
                interval_stay.append(case_ret)

        returns_per_interval = [interval_down, interval_stay, interval_up]

        return returns_per_interval

    def is_one_hot(self, one_hot_vector):
        one_values = 0
        sums=0
        for number in one_hot_vector:
            sums+=number
            if(number==1.0):
                one_values +=1
        is_one_hot = (sums == 1 and one_values == 1)
        return is_one_hot

    def generate_up_soft_label_linear(self, x_percent_avgs, ret): #assumes positive return
        soft_label = [0.0, 0.0, 0.0]

        base_prob_up = 0.5
        extra_prob_up = min((1-base_prob_up)*(ret/x_percent_avgs["up_top_x_percent_avg"]), (1-base_prob_up))

        stay_prob = (1-base_prob_up) - extra_prob_up

        soft_label[2] = base_prob_up + extra_prob_up
        soft_label[1] = stay_prob

        return soft_label


    def generate_down_soft_label_linear(self, x_percent_avgs, ret ): #assumes negative return
        soft_label = [0.0, 0.0, 0.0]

        base_prob_down = 0.5

        extra_prob_down = min((1 - base_prob_down) * (ret / x_percent_avgs["down_low_x_percent_avg"]), (1-base_prob_down))

        stay_prob = (1-base_prob_down) - extra_prob_down

        soft_label[0] = base_prob_down + extra_prob_down
        soft_label[1] = stay_prob

        return soft_label

    def generate_stay_soft_label_linear(self, ret):
        soft_label = [0.0, 0.0, 0.0]

        base_prob_stay = 0.60
        extra_prob_stay = 0

        prob_up = 0
        prob_down = 0

        if(ret < 0):
            extra_prob_stay = min((1 - ret/self.one_hot_interval[0]) * (1-base_prob_stay), 1-base_prob_stay)
            prob_down = (1-base_prob_stay) - extra_prob_stay

        else:
            extra_prob_stay = min((1- ret/self.one_hot_interval[1]) * (1-base_prob_stay), 1-base_prob_stay)
            prob_up = (1 - base_prob_stay) - extra_prob_stay

        prob_stay = base_prob_stay + extra_prob_stay

        soft_label[0] = prob_down
        soft_label[1] = prob_stay
        soft_label[2] = prob_up

        return soft_label


    def find_top_x_avgs(self,returns_per_interval, x_percent):
        averages = {}
        interval_up = copy.deepcopy(returns_per_interval[2])
        interval_down = copy.deepcopy(returns_per_interval[0])
        up_count = 0
        down_count = 0
        interval_up.sort()
        interval_down.sort()
        up_sum = 0
        down_sum = 0
        ninty_percent_index_up = int((len(interval_up)-1) * (1- x_percent))
        ninty_percent_index_down = int(len(interval_down) * x_percent)

        for i in range(ninty_percent_index_up, len(interval_up)):
            up_sum += interval_up[i]
            up_count += 1

        for i in range(0, ninty_percent_index_down):
            down_sum += interval_down[i]
            down_count += 1

        averages["up_top_x_percent_avg"] = float(up_sum/up_count)
        averages["down_low_x_percent_avg"] = float(down_sum/down_count)

        return averages





