
import requests

data = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json').json()
price = round(float(data['bpi']['USD']['rate'].replace(',','')), 2)
print(price)