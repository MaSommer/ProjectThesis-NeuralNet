import numpy as np


class CaseManager():

    def __init__(self, cases, validation_fraction=0.1, test_fraction=0.1):
        self.cases = cases
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.training_fraction = 1 - (validation_fraction+test_fraction)
        self.training_cases = cases
        self.validation_cases = []
        self.testing_cases = []
        #self.organize_cases()

#this method defines what is training_cases, test_cases and validation_cases
    def organize_cases(self):
        ca = np.array(self.cases)
        separator1 = int(round(len(self.cases) * self.training_fraction))
        separator2 = int(separator1 + round(len(self.cases) * self.validation_fraction))
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases