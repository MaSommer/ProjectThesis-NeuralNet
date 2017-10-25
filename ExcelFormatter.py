import xlwt


class ExcelFormatter():

    def __init__(self, hyp_type_1, hyp_type_2, first_run=False):
        self.output("results.csv", "sheet1", hyp_type_1, hyp_type_2)
        self.first_run =  first_run

    def output(self, filename, sheet, hyp_type_1, hyp_type_2):
        book = xlwt.Workbook()

        sh = book.add_sheet(sheet)
        hyp_dicts_type_1 = [{}, {}, {}]
        hyp_dict_type_2 = [{}]

        #self.delete_existing_file(sh)
        col = self.write_hyp_dicts_to_file_type_1(hyp_type_1, sh)
        self.write_hyp_dicts_to_file_type_2(hyp_type_2, sh, col)

        book.save(filename)

    def delete_existing_file(self, worksheet):
        rows_to_move = worksheet.rows[0:]
        for row in rows_to_move:
            new_row_number = row._Row__idx - 2
            row._Row__idx = new_row_number
            for cell in row._Row__cells.values():
                if cell:
                    cell.rowx = new_row_number
            worksheet.rows[new_row_number] = row
        # now delete any remaining rows
        del worksheet.rows[new_row_number + 1:]

    def write_hyp_dicts_to_file_type_1(self, hyp_dicts_type_1, sh):
        col = 0
        for hyp_dict in hyp_dicts_type_1:
            column_labels, values = self.generate_column_labels_and_values(hyp_dict)
            if (self.first_run):
                self.write_to_file(column_labels, sh, 0, col)
            _, col = self.write_to_file(values, sh, 1, col)
        return col

    def write_hyp_dicts_to_file_type_2(self, hyp_dicts_type_2, sh, col):
        for hyp_dict in hyp_dicts_type_2:
            column_labels,values = self.generate_column_labels_and_values(hyp_dict)
            label_index = 0
            if (self.first_run):
                for label in column_labels:
                    if (label_index == 0):
                        start_col = col
                    else:
                        start_col = label_index*len(values[label_index-1])+col

                    for i in range(0, len(values[label_index])):
                        col_data = ""+label+"-stock-"+str(i)
                        sh.write(0, start_col+i, col_data)
                    label_index += 1

            for value in values:
                for i in range(0, len(value)):
                    sh.write(1, col, value[i])
                    col += 1
        return col



    def write_to_file(self, dict, sh, row, col):
        for col_data in dict:
            sh.write(row, col, col_data)
            col += 1
        return row, col

    def generate_column_labels_and_values(self, hyp):
        colum_labels = []
        values = []
        for key in hyp:
            value = hyp[key]
            colum_labels.append(key)
            values.append(value)
        return colum_labels, values


hyp1 = {}
hyp1["ret"] = 2.5
hyp1["ret-up"] = 2
hyp1["rey-down"] = 3.0
hyp1["sd"] = 0.5
hyp1["stock-nr"] = 1
hyp1["stock-drus"] = "pikk"
hyp2 = {}
hyp2["rag"] = 18
hyp2["fag-up"] = 20
hyp2["fag-down"] = 30
hyp2["fitte"] = 24
hyp2["kuken-nr"] = 24.5
hyp2["kusa-drus"] = "dick_size"

hyp3 = {}
hyp3["returns"] = [1, 2, 3, 4, 5]
hyp3["sds"] = [0.5, 0.2, 0.8, 0.9]

hyp_type_1 = [hyp1, hyp2]
hyp_type_2 = [hyp3]

ExcelFormatter(hyp_type_1, hyp_type_2)