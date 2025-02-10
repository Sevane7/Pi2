from BloombergData import BloombergData
from Hurst import HurstDistribution
from Backtest import BacktestStrategy
import warnings

warnings.filterwarnings("ignore")


analyst_name = "Sevane"

ticker = "SPY"
timeframe = "Daily"

def run_tests(analyst_name = analyst_name):

    start = "2010-01-01"
    end = "2015-01-08"


    df = BloombergData(ticker, timeframe, start, end)

    hurst = HurstDistribution(df)

    backtest = BacktestStrategy(hurst, analyst_name, False)


def test_backtest_parameters(start_dates = ["2010-01-01"],
                             end_dates = ["2015-01-08"],
                             h_fq = "3Y",
                             N_generations = [1000],
                             horizons = [3, 5]):
    """Teste différentes combinaisons de paramètres et lève des erreurs en cas de problème."""
    for start in start_dates:

        for end in end_dates:
        
            if start >= end:
                raise ValueError(f"La date de début {start} doit être antérieure à la date de fin {end}.")
            
            for N_generation in N_generations:
        
                if N_generation <= 0:
                    raise ValueError(f"N_generation doit être un entier positif, reçu: {N_generation}")
                
                for horizon in horizons:
        
                    if horizon <= 0:
                        raise ValueError(f"Horizon doit être un entier positif, reçu: {horizon}")
                    
                    try:
                        print(f"Test avec start={start}, end={end}, N_generation={N_generation}, horizon={horizon}")
                        
                        df = BloombergData(ticker, timeframe, start, end)
                        hurst = HurstDistribution(df)
                        backtest = BacktestStrategy(hurst, "Sevane", True)
                        backtest.simple_backtest(h_fq, N_generation, horizon)
                        
                    except Exception as e:
                        print(f"Erreur avec start={start}, end={end}, N_generation={N_generation}, horizon={horizon}:\n {e}")


test_backtest_parameters()

# run_tests()