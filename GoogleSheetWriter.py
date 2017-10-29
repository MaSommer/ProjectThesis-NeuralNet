from __future__ import print_function
import httplib2
import os

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

import requests
import BeautifulSoup
try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None


# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/sheets.googleapis.com-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Google Sheets API Python Quickstart'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'sheets.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def main(hyp_type_1, hyp_type_2, ordered_label_list_type_1, ordered_label_list_type_2):
    """Shows basic usage of the Sheets API.

    Creates a Sheets API service object and prints the names and majors of
    students in a sample spreadsheet:
    https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit
    """
    # url_login = "https://accounts.google.com/ServiceLogin"
    # url_auth = "https://accounts.google.com/ServiceLoginAuth"
    # session = SessionGoogle(url_login, url_auth, "martymasommah@gmail.com")
    # print(session.get("http://plus.google.com"))

    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                    'version=v4')
    service = discovery.build('sheets', 'v4', http=http,
                              discoveryServiceUrl=discoveryUrl)

    spreadsheetId = '1ihqE4vwFsiHE9IMnE2WsCI9MUZki_sIdvrkVniQcvno'
    personal_sheet = find_personal_sheet()
    rangeName_personal = personal_sheet + '!A1'
    rangeName_all = 'all_results!A1'

    label_list = []
    value_list = []

    l1, v1 = write_hyp_dicts_to_file_type_1(hyp_type_1, ordered_label_list_type_1)
    #l2, v2 = write_hyp_dicts_to_file_type_2(hyp_type_2, ordered_label_list_type_2)

    label_list.extend(l1)
    #label_list.extend(l2)
    value_list.extend(v1)
    #value_list.extend(v2)

    write_to_sheet(label_list, value_list, service, spreadsheetId, rangeName_personal)
    write_to_sheet(label_list, value_list, service, spreadsheetId, rangeName_all)



def write_to_sheet(label_list, value_list, service, spreadsheetId, rangeName):

    if (is_already_written_to(service, spreadsheetId, rangeName)):
        values = [
            value_list
        ]
    else:
        values = [
            label_list, value_list
        ]
    body = {
        'values': values
    }
    service.spreadsheets().values().append(
        spreadsheetId=spreadsheetId, range=rangeName,
        valueInputOption="USER_ENTERED", body=body).execute()

def is_already_written_to(service, spreadsheetId, rangeName):
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheetId, range=rangeName).execute()
    values = result.get('values', [])
    if len(values) == 0:
        return False
    else:
        return True


def write_hyp_dicts_to_file_type_1(hyp_dicts_type_1, ordered_label_list_type_1):
    label_list = []
    value_list = []
    for label in ordered_label_list_type_1:
        for hyp_dict in hyp_dicts_type_1:
            if (label in hyp_dict):
                value = hyp_dict[label]
                label_list.append(str(label))
                value_list.append(str(value))
    return label_list, value_list

def write_hyp_dicts_to_file_type_2(hyp_dicts_type_2, ordered_label_list_type_2):
    label_list = []
    value_list = []
    for hyp_dict in hyp_dicts_type_2:
        stock_index_list_with_values = auto_generated_list()

        for i in range(0, len(stock_index_list_with_values)):
            for label in ordered_label_list_type_2:
                if (label in hyp_dict):
                    label_data = "" + label + "-stock-" + str(i)
                    label_list.append(label_data)
        stock_index_list_with_values = auto_generated_list()
        for i in range(0, len(hyp_dict.itervalues().next())):
            stock_info = []
            for label in ordered_label_list_type_2:
                if (label in hyp_dict):
                    values = hyp_dict[label]
                    stock_info.append((values[i]))
            stock_index_list_with_values[int(stock_info[0][1])] = stock_info
        for i in range(0, len(stock_index_list_with_values)):
            stock_info = stock_index_list_with_values[i]
            if isinstance(stock_info, list):
                for j in range(0, len(stock_info)):
                    value_list.append(str(stock_info[j][0]))
            else:
                for j in range(0, len(hyp_dict)):
                    value_list.append(str("NA"))

    return label_list, value_list


def auto_generated_list():
    l = []
    for i in range(0, 100):
        l.append(0)
    return l

def find_personal_sheet():
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    dir = "ProjectThesis-NeuralNet/target/personal_sheet.txt"
    f = open(os.path.join(os.path.abspath(os.path.join(__location__, os.pardir)), dir));
    personal_sheet = f.readline()
    f.close()
    return personal_sheet

# class SessionGoogle:
#     def __init__(self, url_login, url_auth, login, pwd):
#         self.ses = requests.session()
#         # login_html = self.ses.get(url_login)
#         # soup_login = BeautifulSoup(login_html.content).find('form').find_all('input')
#         my_dict = {}
#         # for u in soup_login:
#         #     if u.has_attr('value'):
#         #         my_dict[u['name']] = u['value']
#         #         # override the inputs without login and pwd:
#         my_dict['Email'] = login
#         my_dict['Passwd'] = pwd
#         self.ses.post(url_auth, data=my_dict)
#
#     def get(self, URL):
#         return self.ses.get(URL).text


# hyp1 = {}
# hyp1["ret"] = 2.5
# hyp1["ret-up"] = 2
# hyp1["rey-down"] = 3.0
# hyp1["sd"] = 0.5
# hyp1["stock-nr"] = 1
# hyp1["stock-drus"] = "pikk"
# hyp2 = {}
# hyp2["rag"] = 18
# hyp2["fag-up"] = 20
# hyp2["fag-down"] = 30
# hyp2["fitte"] = 24
# hyp2["kuken-nr"] = 24.5
# hyp2["kusa-drus"] = "dick_size"
#
# hyp3 = {}
# hyp3["returns"] = [1, 2, 3, 4, 5]
# hyp3["sds"] = [0.5, 0.2, 0.8, 0.9]
#
# hyp_type_1 = [hyp1, hyp2]
# hyp_type_2 = [hyp3]
#
# ordered_label_list_type_1 = ["ret", "ret-up", "rey-down", "stock-nr" ,"rag"]
# ordered_label_list_type_2 = ["returns", "sds"]
#
# main(hyp_type_1, hyp_type_2, ordered_label_list_type_1, ordered_label_list_type_2)