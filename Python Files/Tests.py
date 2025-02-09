from BloombergData import BloombergData
from Hurst import HurstDistribution
from Backtest import BacktestStrategy
import warnings

warnings.filterwarnings("ignore")


analyst_name = "Sevane"

ticker = "SPY"
timeframe = "Daily"
start = "2010-01-01"
end = "2015-01-08"

N_generation = 1000
horizon = 3



df = BloombergData(ticker, timeframe, start, end)

hurst = HurstDistribution(df)

backtest = BacktestStrategy(hurst, "1Y", N_generation, horizon)

backtest.backtest()


# print(backtest.original_data)
# print('mean\n', backtest.mean_forecasted_data)

print(f"MSE : {backtest.hit_ratio}")
# print(f"RMSE: {backtest.rmse}")

# backtest.comparaison()