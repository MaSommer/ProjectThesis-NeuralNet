import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials


json_key = json.load(open('client_secret.json')) # json credentials you downloaded earlier
scope = ['https://docs.google.com/spreadsheets/d/1ihqE4vwFsiHE9IMnE2WsCI9MUZki_sIdvrkVniQcvno/edit#gid=0']


#credentials = ServiceAccountCredentials.from_json_keyfile_name(json_key['martymasommah@gmail.com'], scope)

credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)

gc = gspread.authorize(credentials)

sheet = gc.open("Results").sheet1

all_cells = sheet.range('A1:C6')
print all_cells