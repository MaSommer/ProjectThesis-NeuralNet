import os
import time
import re
from collections import defaultdict

class PortolfioInformation:

#Attribute description:
# - op - day to open price - 0
# - cp - day to closing price - 1
# - tv - day to turnover volume - 2
# - hp - day to high price - 3
# - lp - day to low price - 4
# - edd - day to ex dividend day - 5
# - mc - day to market cap - 6
    numberOfAttributes = 7


#attributeDate is a hashMap that maps attribute description to list where index is the day number and value are the value for the selected stocks

    def __init__(self, selectedStocks, attributes, fromDate, toDate, filename):
        self.selectedStocks = selectedStocks
        global numberOfAttributes
        self.createNextCellMap(attributes)
        self.attributeData = defaultdict(list)
        for i in range(len(attributes)):
            self.attributeData[self.attributeIntegerMap[attributes[i]]] = []

        self.fromDate = self.createIntegerOfDate(fromDate)
        self.toDate = self.createIntegerOfDate(toDate)

        self.defineGlobalAttributes()

        self.readFile(filename)


    def readFile(self, filename):
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        dir = "target/" + filename
        f = open(os.path.join(os.path.abspath(os.path.join(__location__, os.pardir)), dir));

        row = 0
        line = f.readline()
        start_time = time.time()
        #iterates over all lines in the .txt file
        while (line != ""):
            rowCells = line.split("\t")
            if (row != 0 and self.checkIfDateIsWithinIntervall(rowCells[0])):
                self.addColDataFromRow(rowCells[1:len(rowCells)])
            if (row%200 == 0):
                print("--- %s seconds ---" % (time.time() - start_time) + " after row " + str(row))
            row+=1
            line = f.readline()

#add all cells from a row to attributeData-hashmap
    def addColDataFromRow(self, rowCells):
        col = 0
        attributeDataForRow = {}
        #-1 because last cell in row is 'endrow'
        while (col < len(rowCells)-1):
            print(col)
            print(len(rowCells))
            if (self.checkIfSelected(col)):
                data = self.getDataPoint(rowCells[col])
                dataType = (col%self.numberOfAttributes)
                print("date: " + str(rowCells[0]) + " col: " + str(col) + " datatype: " + str(dataType) + " data: " + str(data) + " step: " + str(self.nextCellStepMap[col%self.numberOfAttributes]))
                attributeDataForRow.setdefault(dataType, []).append(data)
                col += self.nextCellStepMap[col%self.numberOfAttributes]
                print("col after: " + str(col))
            else:
                col+=self.numberOfAttributes
        for key in attributeDataForRow.keys():
            list = attributeDataForRow[key]
            self.attributeData.setdefault(key, []).append(list)

#returns the float of the number of -1 if it is not a float or a digit
    def getDataPoint(self, rowCell):
        output = self.convertDigitWithoutComma(rowCell)
        if (output != "" and (re.match("^\d+?\.\d+?$", output) is not None or output.isdigit())):
            return float(output)
        else:
            return -1.0

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
        if (date >= self.fromDate and date <= self.toDate):
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


    def defineGlobalAttributes(self):
        self.OPEN_PRICE = 0
        self.CLOSING_PRICE = 1
        self.TURNOVER_VOLUME = 2
        self.HIGH_PRICE = 3
        self.LOW_PRICE = 4
        self.EX_DIV_DAY = 5
        self.MARKET_CAP = 6

#creates the nextCellStepMap which tells how many step the col should move when iterating. It depends on how
#many attributes are selected.
#if only [op, tv] is selected this map returns 1 if col = 0 and returns 6 if col = 1
    def createNextCellMap(self, attributes):
        self.nextCellStepMap = {}
        self.attributeIntegerMap = {}
        self.attributeIntegerMap["op"] = 0
        self.attributeIntegerMap["cp"] = 1
        self.attributeIntegerMap["tv"] = 2
        self.attributeIntegerMap["hp"] = 3
        self.attributeIntegerMap["lp"] = 4
        self.attributeIntegerMap["edd"] = 5
        self.attributeIntegerMap["mc"] = 6
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
        print(self.nextCellStepMap)