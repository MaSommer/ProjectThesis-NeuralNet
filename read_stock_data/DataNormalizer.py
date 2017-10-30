


def normailize_regular(inputPortfolioInformation, attributeDataForRow, previous_attribute_data_for_row,
                       datatype, float_data, selected_stocks, stock_nr):
    data1 = float_data
    if (datatype == inputPortfolioInformation.OPEN_PRICE):
        data2 = previous_attribute_data_for_row[inputPortfolioInformation.CLOSING_PRICE][selected_stocks]

    elif (datatype == inputPortfolioInformation.CLOSING_PRICE or datatype == inputPortfolioInformation.HIGH_PRICE
                  or datatype == inputPortfolioInformation.LOW_PRICE ):
        data2 = attributeDataForRow[inputPortfolioInformation.OPEN_PRICE][selected_stocks]

    elif datatype == inputPortfolioInformation.TURNOVER_VOLUME:
        data2 = inputPortfolioInformation.global_avg_volume
    if (data2 == 0):
        return 0

    normalized_data = float((data1/data2)-1)
    inputPortfolioInformation.update_min_max(normalized_data, datatype, stock_nr)
    return normalized_data


#TODO: implement the following function
def normalize_with_min_max(data, min, max):
    if (max-min == 0):
        return 0
    return (data-min)/(max-min)

def normalize_with_max_and_seperate_neg_and_pos(data, min, max):
    if (max == 0):
        return 0
    if (data < 0):
        if (not (is_float_or_int(data) or is_float_or_int(min))):
            print("Data: " + data + " Min: " + min)
            return 0
        return -data/min
    else:
        if (not (is_float_or_int(data) or is_float_or_int(max))):
            print("Data: " + data + " Max: " + max)

            return 0
        return data/max

def is_float_or_int(data):
    if (data == None):
        return False
    if (isinstance(data, int) or isinstance(data, float)):
        return True
    else:
        return False