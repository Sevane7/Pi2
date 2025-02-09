from BloombergData import BloombergData
from Hurst import HurstDistribution
from Backtest import BacktestStrategy
import warnings

warnings.filterwarnings("ignore")


analyst_name = "Ton Prenom"

ticker = "SPY"
timeframe = "Daily"
start = "2010-01-01"
end = "2015-01-08"


df = BloombergData(ticker, timeframe, start, end)

hurst = HurstDistribution(df)

backtest = BacktestStrategy(hurst, analyst_name)
