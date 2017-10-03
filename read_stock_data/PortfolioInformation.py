import os
import time
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


#data is a hashMap that maps attribute description to list where index is the day number and value are the value for the selected stocks

    def __init__(self, selectedStocks, attributes, fromDate, toDate, filename):
        self.selectedStocks = selectedStocks
        global numberOfAttributes
        self.createNextCellMap(attributes)
        self.attributeData = defaultdict(list)
        for i in range(len(attributes)):
            self.attributeData[self.attributeIntegerMap[attributes[i]]] = []

        self.fromDate = fromDate
        self.toDate = toDate
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
        while (line != ""):
            line = f.readline()
            rowCells = line.split("\t")
            if (row != 0 and self.checkIfDateIsWithinIntervall(rowCells[0])):
                self.addColDataFromRow(rowCells[1:len(rowCells)])
            if (row%200 == 0):
                print("--- %s seconds ---" % (time.time() - start_time) + " after row " + str(row))
            row+=1

    def addColDataFromRow(self, rowCells):
        col = 0
        attributeDataForRow = {}
        while (col < len(rowCells)):
            if (self.checkIfSelected(col)):
                data = rowCells[col]
                dataType = col%self.numberOfAttributes
                attributeDataForRow.setdefault(dataType, []).append(data)
            col+=self.nextCellStepMap[col%self.numberOfAttributes]
        for key in attributeDataForRow.keys():
            list = attributeDataForRow[key]
            self.attributeData.setdefault(key, []).append(list)


    def checkIfSelected(self, col):
        if (self.selectedStocks[col%self.numberOfAttributes] == 0):
            return False
        else:
            return True

    def nextStepSize(self, col):
        colMod = col%self.numberOfAttributes
        return self.nextCellStepMap[colMod]

    def checkIfDateIsWithinIntervall(self, date):
        if (date >= self.fromDate and date <= self.toDate):
            return True
        else:
            return False

    def defineGlobalAttributes(self):
        self.OPEN_PRICE = 0
        self.CLOSING_PRICE = 1
        self.TURNOVER_VOLUME = 2
        self.HIGH_PRICE = 3
        self.LOW_PRICE = 4
        self.EX_DIV_DAY = 5
        self.MARKET_CAP = 6

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