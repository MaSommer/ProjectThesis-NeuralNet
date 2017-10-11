import os.path


def readSelectedStocks(filename):
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dir = "target/selected"+filename
    f = open(os.path.join(os.path.abspath(os.path.join(__location__, os.pardir)), dir));
    line = f.readline()

    outputList = []
    number_of_selected = 0
    while (line != ""):
        outputList.append(int(line))
        if (line == "1"):
            number_of_selected+=1
        line = f.readline()
    return outputList, number_of_selected

