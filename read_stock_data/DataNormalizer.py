import numpy as np



def normailize_regular(inputPortfolioInformation, attributeDataForRow, previous_attribute_data_for_row,
                       datatype, float_data, selected_stocks):
    data1 = float_data
    if (datatype == inputPortfolioInformation.OPEN_PRICE):
        data2 = previous_attribute_data_for_row[inputPortfolioInformation.CLOSING_PRICE][selected_stocks]

    elif (datatype == inputPortfolioInformation.CLOSING_PRICE or datatype == inputPortfolioInformation.HIGH_PRICE
                  or datatype == inputPortfolioInformation.LOW_PRICE ):
        data2 = attributeDataForRow[inputPortfolioInformation.OPEN_PRICE][selected_stocks]

    elif datatype == inputPortfolioInformation.TURNOVER_VOLUME:
        data2 = inputPortfolioInformation.global_avg_volume

    return float((data1/data2)-1)


#TODO: implement the following function
def normalize_with_min_and_max_value(data, min, max):
    return (data-min)/(max-min)