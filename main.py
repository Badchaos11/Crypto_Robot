import pandas as pd
from binance import ThreadedWebsocketManager, Client
import logging
import secrets
from secrets import client
from binance.exceptions import BinanceAPIException


symbol = 'BTCUSDT'


def handler_message(msg):
    time = pd.to_datetime(msg['E'], unit='ms')
    price = round(float(msg['c']), 2)
    print('"Time: {} | Price: {}"'.format(time, price))
    price = int(round(price))

    if price % 10 == 0:
        try:
            order = client.create_order(symbol='BTCUSDT', side='BUY', type='MARKET', quantity=0.005)
            print("\n" + 50 * "-")
            print("I {} BTC for {} USD".format(order['executedQty'], order['cummulativeQuoteQty']))
            print(50 * "-")
            twm.stop()
        except BinanceAPIException:
            print("Account has insufficient balance for requested action. I bought BTC, but error didn't showed it")
            twm.stop()



if __name__ == '__main__':

    #logger = logging.getLogger('websockets')
    #logger.setLevel(logging.DEBUG)
    #logger.addHandler(logging.StreamHandler())

    print('started')

    client.get_account()

    twm = ThreadedWebsocketManager(api_key=secrets.api_key, api_secret=secrets.api_secret)
    twm.start()
    twm.start_symbol_miniticker_socket(callback=handler_message, symbol=symbol)
    twm.join()
