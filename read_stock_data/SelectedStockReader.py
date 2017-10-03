import os.path


def readSelectedStocks(filename):
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dir = "target/selected"+filename
    f = open(os.path.join(os.path.abspath(os.path.join(__location__, os.pardir)), dir));
    line = f.readline()

    outputList = []
    c = 1
    while (line != ""):
        outputList.append(int(line))
        line = f.readline()
    return outputList

