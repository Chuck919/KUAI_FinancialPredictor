import pandas as pd


predictions = pd.read_csv('predictions.csv')

filtered = predictions[((predictions['bnb_predictions'] == 0) & (predictions['svc_predictions'] == 0)) | ((predictions['bnb_predictions'] == 4) & (predictions['svc_predictions'] == 4))]


print(filtered)



percentage = (filtered['bnb_predictions'].value_counts(normalize=True) * 100).to_dict()
print("Positive sentiment percentage:", percentage[4])
print("Negative sentiment percentage:", percentage[0])

majority = filtered['bnb_predictions'].value_counts()
# Check the majority in the filtered dataframe
if majority.idxmax() == 4:
    signal = 'BUY'
    print("Models predict: Buy")
elif majority.idmax() == 0:
    signal = 'SELL'
    print("Models predict: Sell")
else:
    signal = None
    print("Equal amount of positive and negative sentiment")
    

#Code to implement Binance.US connection and place buy and sell orders based on predictions
#Requires Binance API key and API Secret

'''
from binance.client import Client
from decimal import Decimal, ROUND_DOWN

api_key = 'Your API Key'
api_secret = 'Your API Secret'
client = Client(api_key, api_secret, tld='us')

if signal == 'BUY':
    buy_order = client.order_market_buy(symbol = 'BTCUSDT', quantity = 0.001) #you can replace symbol with which crypto you are making predictions for, for instance this would be the Bitcoin US Tether trading pair
    print('Bought')
elif signal == 'SELL':
    sell_order = client.order_market_sell(symbol = 'BTCUSDT', quantity = 0.001) #also replace quantity with how much you want to buy in terms of the Crypto, for instance this would be 0.001 Bitcoin
    print('Sold')
else:
    pass
'''