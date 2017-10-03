import SelectedStockReader as ssr
import PortfolioInformation as pi

def Main():
    fromDate = "02.02.2007"
    toDate = "10.08.2017"
    attributes = ["op", "cp", "tv"]
    selectedSP500 = ssr.readSelectedStocks("S&P500.txt")
    sp500 = pi.PortolfioInformation(selectedSP500, attributes, fromDate, toDate, "S&P500.txt")
    print(sp500.attributeData)

Main()