import threading
import time
import BinancePrice
from datetime import datetime
from BinanceApi import BinanceFutureApi

from config.config import get_config

DECIMAL_PLACES = 3
output_log_path = 'output_log.log'


# GLOBAL
gPrice = {}
gInfoTradingUp = {}
gInfoTradingDown = {}
exchange = None


def log(data):
    with open(output_log_path, 'a+') as f:
        f.write(str(datetime.now()) + ' ' + str(data) + '\n')


############################################################################
# Internal
def list_buy_price(start, bottom, delta):
    list_output = []
    try:
        i = 0
        while(start > bottom + i*delta):
            if bottom + i*delta < start:
                list_output.insert(
                    0, round((bottom + i*delta), DECIMAL_PLACES))
                i += 1
    except Exception as e:
        log("list_buy_price: "+str(e))
    list_output = list(dict.fromkeys(list_output))
    return list_output


def calculateSellPrice(enter, profit):
    return round(enter+profit, DECIMAL_PLACES)


def list_sell_price(start, top, delta):
    list_output = []
    try:
        i = 0
        while(start < top - i*delta):
            if top - i*delta > start:
                list_output.insert(
                    0, round((top - i*delta), DECIMAL_PLACES))
                i += 1
    except Exception as e:
        log("list_sell_price: "+str(e))
    list_output = list(dict.fromkeys(list_output))
    return list_output


def calculateBuyPrice(enter, profit):
    return round(enter-profit, DECIMAL_PLACES)


############################################################################
# Thread


def get_list_price():
    global gPrice
    global gInfoTradingUp
    global gInfoTradingDown
    try:

        # 1. Get list buy
        if 'price' in gPrice and gPrice['price']:
            list_buy_prices = list_buy_price(gPrice['price'], BOTTOM, DELTA)
            # print(list_buy_prices)

            # 2. Update list buy info
            for price in list_buy_prices:
                if str(price) not in gInfoTradingUp:
                    print(
                        f"-- gInfoTradingUp Add new price for Trade {str(price)}")
                    # initial trading info
                    gInfoTradingUp[str(price)] = {}
                    gInfoTradingUp[str(price)]['buyPrice'] = price
                    gInfoTradingUp[str(price)]['sellPrice'] = calculateSellPrice(
                        price, PROFIT)
                    gInfoTradingUp[str(price)]['buyId'] = 0
                    gInfoTradingUp[str(price)]['sellId'] = 0
                    gInfoTradingUp[str(price)]['buyStatus'] = None
                    gInfoTradingUp[str(price)]['sellStatus'] = None
                    gInfoTradingUp[str(price)]['isTrading'] = False

            # 3. Remove
            if len(list_buy_prices) < len(gInfoTradingUp):
                for price_info in gInfoTradingUp:
                    if float(price_info) not in list_buy_prices:
                        if not gInfoTradingUp[str(price_info)]['isTrading']:
                            print(
                                f"-- gInfoTradingUp REMOVE price: {str(price_info)}")
                            del gInfoTradingUp[str(price_info)]

            # print(gInfoTradingUp)

            list_sell_prices = list_sell_price(gPrice['price'], TOP, DELTA)
            # print(list_sell_prices)

            # 2. Update list buy info
            for price in list_sell_prices:
                if str(price) not in gInfoTradingDown:
                    print(
                        f"-- gInfoTradingDown Add new price for Trade {str(price)}")
                    # initial trading info
                    gInfoTradingDown[str(price)] = {}
                    gInfoTradingDown[str(price)]['sellPrice'] = price
                    gInfoTradingDown[str(price)]['buyPrice'] = calculateBuyPrice(
                        price, PROFIT)
                    gInfoTradingDown[str(price)]['sellId'] = 0
                    gInfoTradingDown[str(price)]['buyId'] = 0
                    gInfoTradingDown[str(price)]['sellStatus'] = None
                    gInfoTradingDown[str(price)]['buyStatus'] = None
                    gInfoTradingDown[str(price)]['isTrading'] = False

            # 3. Remove
            if len(list_sell_prices) < len(gInfoTradingDown):
                for price_info in gInfoTradingDown:
                    if float(price_info) not in list_sell_prices:
                        if not gInfoTradingDown[str(price_info)]['isTrading']:
                            print(
                                f"-- gInfoTradingDown  REMOVE price: {str(price_info)}")
                            del gInfoTradingDown[str(price_info)]

    except Exception as e:
        print("get_list_price: "+str(e))
    t = threading.Timer(0.5, get_list_price)
    t.start()


countPrint = 0


def trading_up():
    global gPrice
    global gInfoTradingUp
    global countPrint
    try:
        countPrint += 1
        trades_status = exchange.fetch_orders(COIN)
        for price in gInfoTradingUp:
            # 1. check if not start then start limit order
            if not gInfoTradingUp[price]['isTrading']:
                if gPrice['price'] < gInfoTradingUp[price]['buyPrice'] + DELTA:
                    print(
                        f"Start buy for the price {gInfoTradingUp[price]['buyPrice']}")
                    # buy order
                    buy_id, buy_amount, enter_price = exchange.create_order(
                        COIN, 'limit', 'buy', QUANTITY, entry_price=gInfoTradingUp[price]['buyPrice'])
                    print(
                        f'BUY --- id: {buy_id}, amount: {buy_amount}, price: {enter_price}')
                    log(f'BUY --- id: {buy_id}, amount: {buy_amount}, price: {enter_price}')
                    if buy_id != 0:
                        gInfoTradingUp[price]['buyId'] = str(buy_id)
                        gInfoTradingUp[price]['isTrading'] = True

            if gInfoTradingUp[price]['isTrading'] == True:
                # update buy status
                if gInfoTradingUp[price]['buyId'] != 0:
                    if not countPrint % 10:
                        print(
                            f"INFO: BUY id: {gInfoTradingUp[price]['buyId']}, price: {gInfoTradingUp[price]['buyPrice']}, status: {gInfoTradingUp[price]['buyStatus']}")

                if gInfoTradingUp[price]['buyId'] != 0 and \
                        gInfoTradingUp[price]['buyId'] in trades_status and \
                        gInfoTradingUp[price]['buyStatus'] != trades_status[gInfoTradingUp[price]['buyId']]:

                    gInfoTradingUp[price]['buyStatus'] = trades_status[gInfoTradingUp[price]['buyId']]

                # update sell status
                if gInfoTradingUp[price]['sellId'] != 0 and \
                        gInfoTradingUp[price]['sellId'] in trades_status and  \
                        gInfoTradingUp[price]['sellStatus'] != trades_status[gInfoTradingUp[price]['sellId']]:

                    gInfoTradingUp[price]['sellStatus'] = trades_status[gInfoTradingUp[price]['sellId']]

                # clear all if sold
                if gInfoTradingUp[price]['sellId'] != 0 and gInfoTradingUp[price]['sellStatus'] == "closed":
                    # gInfoTradingUp[str(price)]['id'] = 0
                    gInfoTradingUp[price]['buyId'] = 0
                    gInfoTradingUp[price]['sellId'] = 0
                    gInfoTradingUp[price]['buyStatus'] = None
                    gInfoTradingUp[price]['sellStatus'] = None
                    gInfoTradingUp[price]['isTrading'] = False

            # 2. check sell limit.
            # condition: buy has filled, and not sell yet
            if gInfoTradingUp[price]['isTrading'] == True and \
                    gInfoTradingUp[price]['buyStatus'] == "closed" and \
                    gInfoTradingUp[price]['sellId'] == 0:
                sell_id, sell_amount, sell_price = exchange.create_order(
                    COIN, 'limit', 'sell', QUANTITY, entry_price=gInfoTradingUp[price]['sellPrice'])
                print(
                    f'SELL: id: {sell_id}, amount: {sell_amount}, price: {sell_price}')
                log(f'SELL: id: {sell_id}, amount: {sell_amount}, price: {sell_price}')
                if sell_id != 0:
                    gInfoTradingUp[price]['sellId'] = str(sell_id)

    except Exception as e:
        print("trading_up: "+str(e))
    t = threading.Timer(1, trading_up)
    t.start()


countPrint2 = 0


def trading_down():
    global gPrice
    global gInfoTradingDown
    global countPrint2
    try:
        countPrint2 += 1
        trades_status = exchange.fetch_orders(COIN)
        for price in gInfoTradingDown:
            # 1. check if not start then start sell limit order
            if not gInfoTradingDown[price]['isTrading']:
                if gPrice['price'] > gInfoTradingDown[price]['sellPrice'] - DELTA:
                    print(
                        f"Start sell for the price {gInfoTradingDown[price]['sellPrice']}")
                    # sell order
                    sell_id, sell_amount, enter_price = exchange.create_order(
                        COIN, 'limit', 'sell', QUANTITY, entry_price=gInfoTradingDown[price]['sellPrice'])
                    print(
                        f'SELL - id: {sell_id}, amount: {sell_amount}, price: {enter_price}')
                    log(f'SELL - id: {sell_id}, amount: {sell_amount}, price: {enter_price}')
                    if sell_id != 0:
                        gInfoTradingDown[price]['sellId'] = str(sell_id)
                        gInfoTradingDown[price]['isTrading'] = True

            if gInfoTradingDown[price]['isTrading'] == True:
                # update sell status
                if gInfoTradingDown[price]['sellId'] != 0:
                    if not countPrint2 % 10:
                        print(
                            f"SELL id: {gInfoTradingDown[price]['sellId']}, price: {gInfoTradingDown[price]['sellPrice']}, status: {gInfoTradingDown[price]['sellStatus']}")

                if gInfoTradingDown[price]['sellId'] != 0 and \
                        gInfoTradingDown[price]['sellId'] in trades_status and \
                        gInfoTradingDown[price]['sellStatus'] != trades_status[gInfoTradingDown[price]['sellId']]:

                    gInfoTradingDown[price]['sellStatus'] = trades_status[gInfoTradingDown[price]['sellId']]

                # update buy status
                if gInfoTradingDown[price]['buyId'] != 0 and \
                        gInfoTradingDown[price]['buyId'] in trades_status and  \
                        gInfoTradingDown[price]['buyStatus'] != trades_status[gInfoTradingDown[price]['buyId']]:

                    gInfoTradingDown[price]['buyStatus'] = trades_status[gInfoTradingDown[price]['buyId']]

                # clear all if bought
                if gInfoTradingDown[price]['buyId'] != 0 and gInfoTradingDown[price]['buyStatus'] == "closed":
                    # gInfoTradingDown[str(price)]['id'] = 0
                    gInfoTradingDown[price]['sellId'] = 0
                    gInfoTradingDown[price]['buyId'] = 0
                    gInfoTradingDown[price]['sellStatus'] = None
                    gInfoTradingDown[price]['buyStatus'] = None
                    gInfoTradingDown[price]['isTrading'] = False

            # 2. check buy limit.
            # condition: buy has filled, and not sell yet
            if gInfoTradingDown[price]['isTrading'] == True and \
                    gInfoTradingDown[price]['sellStatus'] == "closed" and \
                    gInfoTradingDown[price]['buyId'] == 0:
                buy_id, buy_amount, buy_price = exchange.create_order(
                    COIN, 'limit', 'buy', QUANTITY, entry_price=gInfoTradingDown[price]['buyPrice'])
                print(
                    f'BUY: id: {buy_id}, amount: {buy_amount}, price: {buy_price}')
                log(f'BUY: id: {buy_id}, amount: {buy_amount}, price: {buy_price}')
                if buy_id != 0:
                    gInfoTradingDown[price]['buyId'] = str(buy_id)

    except Exception as e:
        print("trading_down: "+str(e))
    t = threading.Timer(1, trading_down)
    t.start()


def update_price():
    global gPrice
    gPrice = BinancePrice.coin_price
    t = threading.Timer(0.5, update_price)
    t.start()


############################################################################
# Main
if __name__ == '__main__':
    # get config
    params = get_config('binance.ini', 'binance_future')
    APIKEY = params['apikey']
    SECRET = params['secret']

    # CONFIG
    TOP = float(params['top'])
    BOTTOM = float(params['bottom'])
    DELTA = float(params['delta'])
    PROFIT = float(params['profit'])

    QUANTITY = float(params['quantity'])
    LEVERAGE = int(float(params['leverage']))
    COIN = params['coin']

    web_socket_thread = threading.Thread(
        target=BinancePrice.price_stream, args=(COIN.replace('/', ''), True,))
    web_socket_thread.daemon = True
    web_socket_thread.start()

    exchange = BinanceFutureApi(apikey=APIKEY, secret=SECRET, realnet=True)
    exchange.set_leverage(COIN, LEVERAGE)
    exchange.set_marginType(COIN, margin_type='CROSS')

    ths = []
    # start thread update get_list_buy
    th1 = threading.Thread(target=get_list_price)
    ths.append(th1)
    th1.start()

    # start thread update update_price
    th2 = threading.Thread(target=update_price)
    ths.append(th2)
    th2.start()

    # start thread update trading_up
    # th3 = threading.Thread(target=trading_up)
    # ths.append(th3)
    # th3.start()

    # # start thread update trading_down
    # th4 = threading.Thread(target=trading_down)
    # ths.append(th4)
    # th4.start()

    i = 0

    gInfo = {}
    # just do some update
    while(1):
        i += 1
        time.sleep(1)
        if not i % 50:
            print(gPrice)
            i = 0

        # if i > 10:
        for info in (exchange.get_all_open_positions()):
            # print(datetime.now().strftime('%b %d - '), info)
            if info['pair'] in gInfo:
                if 'maxProfit' in gInfo[info['pair']]:
                    if gInfo[info['pair']]['maxProfit'] < info['unRealizedProfit']:
                        gInfo[info['pair']]['maxProfit'] = info['unRealizedProfit']
                else:
                    gInfo[info['pair']]['maxProfit'] = info['unRealizedProfit']
                gInfo[info['pair']]['current'] = info['unRealizedProfit']
            else:
                gInfo[info['pair']] = {}
                gInfo[info['pair']]['pair'] = info['pair']
            # print(datetime.now().strftime('%b %d - '), gInfo)
        for data, data_info in gInfo.items():
            print(datetime.now().strftime('%b %d - '), data, data_info)
            if 'maxProfit' in data_info:
                if data_info['maxProfit'] > 6:
                    if data_info['current'] <= 0.8*data_info['maxProfit']:
                        exchange.close_position(data_info['pair'])

        # print(exchange.get_balance())
