import os
import time
import re
from collections import defaultdict
import DataNormalizer as normalizer
import sys

class InputPortolfioInformation:

#Attribute description:
# - op - day to open price - 0
# - cp - day to closing price - 1
# - tv - day to turnover volume - 2
# - hp - day to high price - 3
# - lp - day to low price - 4
# - edd - day to ex dividend day - 5
# - mc - day to market cap - 6

#one_hot_vector_interval keeps [low, high] of returns which categorizes the return as no change
#attributeDate is a hashMap that maps attribute description to list where index is the day number and value are the value for the selected stocks

    def __init__(self, selectedStocks, attributes, fromDate, filename, number_of_attributes, number_of_trading_days, one_hot_vector_interval=[0,0], is_output=False, start_time=time.time()):
        self.is_output = is_output
        self.start_time = start_time
        self.selectedStocks = selectedStocks
        self.numberOfAttributes = number_of_attributes
        self.attributes = attributes
        self.createNextCellMap(attributes)
        self.portfolio_data = defaultdict(list)
        self.one_hot_vector_interval = one_hot_vector_interval
        for i in range(len(attributes)):
            self.portfolio_data[self.attributeIntegerMap[attributes[i]]] = []

        self.fromDate = self.createIntegerOfDate(fromDate)
        self.number_of_trading_days = number_of_trading_days

        self.defineGlobalAttributes()

        self.readFile(filename)


    def readFile(self, filename):
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        dir = "target/" + filename
        f = open(os.path.join(os.path.abspath(os.path.join(__location__, os.pardir)), dir));

        row = 0
        line = f.readline()
        #iterates over all lines in the .txt file
        previous_attribute_data_for_row = {}
        #this method makes sure that the output dates are being read two days after input
        #   - one day because input use returns from the day before
        #   - and one day because returns are compared to the input the day before
        skipedDatesForOutput = 0
        self.number_of_days_included = 0
        while (line != ""):
            rowCells = line.split("\t")
            # last statement is to make sure that we don't include the first date for output values
            if (row != 0 and self.checkIfDateIsWithinIntervall(rowCells[0])):
                if (skipedDatesForOutput < 2 and self.is_output):
                    skipedDatesForOutput += 1
                    row += 1
                    line = f.readline()
                    continue
                previous_attribute_data_for_row = self.addColDataFromRow(rowCells[1:len(rowCells)], previous_attribute_data_for_row)

            if (row%500 == 0):
                print("--- Row " + str(row) + " takes \t %s seconds ---" % (time.time() - self.start_time))
            row+=1
            line = f.readline()

#add all cells from a row to attributeData-hashmap
    def addColDataFromRow(self, rowCells, previous_attribute_data_for_row):
        col = 0
        stock_nr = 0
        selected_stocks = 0
        attributeDataForRow = {}
        normalized_data = {}
        #-1 because last cell in row is 'endrow'
        while (col < len(rowCells)):
            if (self.checkIfSelected(col)):
                if (self.is_output):
                    float_data = self.getDataPoint(rowCells[col])
                    one_hot_data = self.generate_one_hot_vector(float_data)
                    dataType = (col%self.numberOfAttributes)
                    for i in range(0, len(one_hot_data)):
                        attributeDataForRow.setdefault(dataType, []).append(one_hot_data[i])
                # if working with the input file
                else:
                    float_data = self.getDataPoint(rowCells[col])
                    dataType = (col%self.numberOfAttributes)
                    self.update_avg_min_and_max_for_datatype(float_data, dataType)
                    attributeDataForRow.setdefault(dataType, []).append(float_data)
                    # checking if prev is not empty, it will be empty the first round
                    if (previous_attribute_data_for_row):
                        normalized_data_point = normalizer.normailize_regular(self, attributeDataForRow, previous_attribute_data_for_row,
                                                      dataType, float_data, selected_stocks)
                        normalized_data.setdefault(dataType, []).append(normalized_data_point)
                col += self.nextCellStepMap[col%self.numberOfAttributes]
                #checks if the iteration over this stock is over and we are ready for a new stock
                if (int(col/self.numberOfAttributes) != stock_nr):
                    selected_stocks += 1
            else:
                col+=self.numberOfAttributes
            stock_nr = int(col/self.numberOfAttributes)
        if (self.is_output):
            self.number_of_days_included += 1
            for key in attributeDataForRow.keys():
                list1 = attributeDataForRow[key]
                self.portfolio_data.setdefault(key, []).append(list1)
        else:
            if (previous_attribute_data_for_row):
                self.number_of_days_included+=1
            for key in normalized_data.keys():
                list1 = normalized_data[key]
                #list1 = attributeDataForRow[key]
                self.portfolio_data.setdefault(key, []).append(list1)

        return attributeDataForRow

    def addColDataFromRow2(self, rowCells):
        col = 0
        attributeDataForRow = {}
        #-1 because last cell in row is 'endrow'
        while (col < len(rowCells)):
            if (self.checkIfSelected(col)):
                if (self.is_output):
                    float_data = self.getDataPoint(rowCells[col])
                    one_hot_data = self.generate_one_hot_vector(float_data)
                    dataType = (col%self.numberOfAttributes)
                    for i in range(0, len(one_hot_data)):
                        attributeDataForRow.setdefault(dataType, []).append(one_hot_data[i])
                else:
                    float_data = self.getDataPoint(rowCells[col])
                    dataType = (col%self.numberOfAttributes)
                    attributeDataForRow.setdefault(dataType, []).append(float_data)
                col += self.nextCellStepMap[col%self.numberOfAttributes]
            else:
                col+=self.numberOfAttributes
        for key in attributeDataForRow.keys():
            list = attributeDataForRow[key]
            self.portfolio_data.setdefault(key, []).append(list)

    def update_avg_min_and_max_for_datatype(self, float_data, datatype):
        if (self.global_data_max[datatype] < float_data):
            self.global_data_max[datatype] = float_data
        if (self.global_data_min > float_data):
            self.global_data_min = float_data
        if (datatype == self.TURNOVER_VOLUME):
            total = self.global_avg_conuter*self.global_avg_volume
            self.global_avg_conuter += 1
            self.global_avg_volume = total/self.global_avg_conuter

#returns one_hot_vector [decrease, no change, increase]
    def generate_one_hot_vector(self, data):
        if (data < self.one_hot_vector_interval[0]):
            return [1, 0, 0]
        elif (data > self.one_hot_vector_interval[1]):
            return [0, 0, 1]
        else:
            return [0, 1, 0]

#returns the float of the number of 0.0 if it is not a float or a digit
#TODO: concider not using 0.0 as value when NA. Check out encode-decoder framework!
    def getDataPoint(self, rowCell):
        output = self.convertDigitWithoutComma(rowCell)
        if (output != "" and (re.match("^\d+?\.\d+?$", output) is not None or output.isdigit())):
            return float(output)
        elif(len(output) > 0 and output[0] == "-"):
            if (output[1:len(output)] != "" and (re.match("^\d+?\.\d+?$", output[1:len(output)]) is not None or output[1:len(output)].isdigit())):
                return float(output)
        else:
            return 0.0

#checks if the cell is a part of the selected stock
    def checkIfSelected(self, col):
        if (self.selectedStocks[int(col/self.numberOfAttributes)] == 0):
            return False
        else:
            return True

#returns the number of steps that col should jump. It depends on how many attrbutes are selected
#if only [op, tv] is selected this method returns 1 if col = 0 and 6 if col = 1
    def nextStepSize(self, col):
        colMod = col%self.numberOfAttributes
        return self.nextCellStepMap[colMod]

#checks if date is between fromdate and todate
    def checkIfDateIsWithinIntervall(self, date):
        date = self.createIntegerOfDate(date)
        if (date >= self.fromDate and self.number_of_days_included < self.number_of_trading_days):
            return True
        else:
            return False

#convert a date of the form dd.mm.yyyy to digit yyyymmdd
    def createIntegerOfDate(self, date):
        output = date.split('.')
        digit = ""
        for i in range(len(output)-1,-1,-1):
            digit += output[i]
        return int(digit)

#removes comma from the digit and replaces it with dot
    def convertDigitWithoutComma(self, input):
        output = ""
        for c in input:
            if (c == ','):
                output += '.'
            else:
                output +=c
        return output


    def initialize_global_min_and_max(self):
        # maps data type to the min and max that is found
        self.global_data_min = {}
        self.global_data_max = {}
        self.global_avg_volume = 0
        #conuts how many already calculated in the averages
        self.global_avg_conuter = 0
        for attribute in self.attributes:
            key = self.attributeIntegerMap[attribute]
            self.global_data_max[key] = -float("inf")
            self.global_data_min[key] = float("inf")

    def defineGlobalAttributes(self):
        self.OPEN_PRICE = 0
        self.CLOSING_PRICE = 1
        self.TURNOVER_VOLUME = 2
        self.HIGH_PRICE = 3
        self.LOW_PRICE = 4
        self.EX_DIV_DAY = 5
        self.MARKET_CAP = 6
        self.initialize_global_min_and_max()

#creates the nextCellStepMap which tells how many step the col should move when iterating. It depends on how
#many attributes are selected.
#if only [op, tv] is selected this map returns 1 if col = 0 and returns 6 if col = 1
    def createNextCellMap(self, attributes):
        self.nextCellStepMap = {}
        self.attributeIntegerMap = {}
        if (not self.is_output):
            self.attributeIntegerMap["op"] = 0
            self.attributeIntegerMap["cp"] = 1
            self.attributeIntegerMap["tv"] = 2
            self.attributeIntegerMap["hp"] = 3
            self.attributeIntegerMap["lp"] = 4
            self.attributeIntegerMap["edd"] = 5
            self.attributeIntegerMap["mc"] = 6
        else:
            for i in range (0, len(attributes)):
                self.attributeIntegerMap[attributes[i]] = i

        attInteger = self.attributeIntegerMap[attributes[0]]
        nextAttInteger = -1
        for i in range(1,len(attributes)):
            nextAttInteger = self.attributeIntegerMap[attributes[i]]
            self.nextCellStepMap[attInteger] = (nextAttInteger-attInteger)
            attInteger = nextAttInteger
        if (nextAttInteger != -1):
            self.nextCellStepMap[attInteger] = (self.numberOfAttributes - self.attributeIntegerMap[attributes[len(attributes)-1]])
        else:
            self.nextCellStepMap[attInteger] = self.numberOfAttributes
