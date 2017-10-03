import SelectedStockReader as ssr
import PortfolioInformation as pi

def Main():
    fromDate = "10.08.2017"
    toDate = "11.08.2017"
    attributes = ["op", "cp", "tv"]
    selectedSP500 = ssr.readSelectedStocks("S&P500v2.txt")
    sp500 = pi.PortolfioInformation(selectedSP500, attributes, fromDate, toDate, "S&P500.txt")
    print(sp500.attributeData)

Main()