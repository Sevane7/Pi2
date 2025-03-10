import os
import pandas as pd
import warnings
from Hurst import apply_hurst
from Forecasting import single_forecast, save_forecast, load_forecast
from Backtest import BacktestStrategy
from Metrics import metrics_strategy


def pipeline_from_data_to_metrics(timeframe : str, ticker : str, dir_path_data = "Data\\Backtest") -> None:

    # Chemin du Dossier Data\TimeFrame contenant les data
    dir_tframe_path = os.path.join(dir_path_data, timeframe)
    output_file = rf"Data\\Backtest\\{timeframe}\\{ticker}.xlsx"
    dir_forecast = rf"Data\\Forecasting\\Covariance Based\\Single Forecast\\{timeframe}"
    hurst_file = f"{ticker} Hurst.xlsx"
    forecast_file = f"{ticker} Forecast.json"
    json_file = f"{ticker}.json"
    xls_file = f"{ticker}.xlsx"
    hurst_path = os.path.join(dir_tframe_path, hurst_file) 

    if xls_file in os.listdir(dir_path_data):
        print(f"{ticker} déjà exécuté.")
        return None

    # Retourne la dataframe contenant les distribution de Hurst
    df : pd.DataFrame = None
    
    if hurst_file in os.listdir(dir_tframe_path):
        df = pd.read_excel(hurst_path, index_col=0)
    
    else:
        df = apply_hurst(ticker=ticker, timeframe=timeframe)
        df.to_excel(hurst_path,index=True)

    sizeMat_horizons = {
            10 : [1, 2],
            30 : [1, 2, 3],
            50 : [1, 3, 5]
        }
    
    # Les fichiers 1min sont trop volumineux, on limite à 6000 lignes
    if(timeframe=="1min"):
        try:
            df=df.tail(6000)
        except NameError as e:
            print(ticker, " : Non exécuté.")
            print(e)
            return None
    
    # Fihier et dossiers pour les forecast
    dic_forecast : dict[int, dict[str,dict[int,pd.DataFrame]]] = {}

    # Vérifie si le forecast exist déjà 
    if json_file in os.listdir(dir_forecast):

        path_forecast = os.path.join(dir_forecast, json_file)

        dic_forecast = load_forecast(path_forecast)

    elif forecast_file in os.listdir(dir_tframe_path):

        path_forecast = os.path.join(dir_tframe_path, forecast_file)

        dic_forecast = load_forecast(path_forecast)
        
    
    # Sinon appliquer le forecast
    else:

        for i, horizons in sizeMat_horizons.items():           

            print(f"Covariance Matrix Size : {i}")
            
            frcst = single_forecast(df, horizons, i)

            dic_forecast[i] = frcst

        forecast_path = os.path.join(dir_tframe_path, forecast_file)

        save_forecast(forecast_path, dic_forecast)

    # Initialiser les variables avant le backtest
    best_size_mat, best_hurst, best_horizon = 0, "", 1
    best_hit, best_forecast = 0, pd.DataFrame(None)

    # Fait les backtest pour déterminer le hit ratio 
    for size_mat in dic_forecast.keys():
        for h_fq in dic_forecast[size_mat].keys():
            for horizon in dic_forecast[size_mat][h_fq].keys():
                df_forecast : pd.DataFrame = dic_forecast[size_mat][h_fq][horizon]
                try:
                    backtestobj = BacktestStrategy(ticker=ticker,
                                                timeframe=timeframe,
                                                h_freq=h_fq,
                                                horizon=horizon,
                                                forecast=df_forecast)
                    
                    if backtestobj.hit_ratio > best_hit:
                        best_size_mat = size_mat
                        best_hurst = h_fq
                        best_horizon = horizon
                        best_hit = backtestobj.hit_ratio
                        best_forecast = df_forecast
                    
                except Exception as e:
                    print(e)
                    print(size_mat, h_fq, horizon)
    
    print(f"{ticker} Hit ratio max : {round(best_hit, 5)},\nHurt frequence {best_hurst},\nSize Matrix : {best_size_mat},\nHorizon : {best_horizon}")

    # Conserver les colonnes nécessaires
    res = df.merge(best_forecast["Forecast"], left_index=True, right_index=True, how="inner")
    hurst_cols = [c for c in df.columns if c.startswith("Hurst")]
    hurst_cols.remove(f"Hurst {best_hurst}")
    res.drop(columns=hurst_cols, inplace=True) 

    # Calculer les metrics
    metrics_strategy(res)

    res.to_excel(output_file, index=True)
    


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    
    timeframe = "Daily"
    ticker = "BNP"
    output_file = rf"Data\\Backtest\\{timeframe}\\{ticker}.xlsx"

    pipeline_from_data_to_metrics(timeframe=timeframe, ticker=ticker)

    pass