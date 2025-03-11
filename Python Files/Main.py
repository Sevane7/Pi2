import os
import pandas as pd
import warnings
from Hurst import apply_hurst
from Forecasting import single_forecast, save_forecast, load_forecast
from Backtest import BacktestStrategy
from Metrics import metrics_strategy, plot_compare_metrics


def pipeline_from_data_to_metrics(timeframe : str, ticker : str, dir_path_data = "Data\\Backtest") -> pd.DataFrame:

    # Chemin du Dossier Data\TimeFrame contenant les data
    dir_tframe_path = os.path.join(dir_path_data, timeframe)
    output_file = rf"Data\\Backtest\\{timeframe}\\{ticker}.xlsx"
    dir_forecast = rf"Data\\Forecasting\\Covariance Based\\Single Forecast\\{timeframe}"
    hurst_file = f"{ticker} Hurst.xlsx"
    forecast_file = f"{ticker} Forecast.json"
    json_file = f"{ticker}.json"
    hurst_path = os.path.join(dir_tframe_path, hurst_file) 


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
        dic_forecast = load_forecast(forecast_path)

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


    # Conserver les colonnes nécessaires
    res = df.merge(best_forecast["Forecast"].to_frame(), left_index=True, right_index=True, how="inner")
    hurst_cols = [c for c in df.columns if c.startswith("Hurst")]
    hurst_cols.remove(f"Hurst {best_hurst}")
    res.drop(columns=hurst_cols, inplace=True) 

    # Calculer les metrics
    metrics_strategy(res)
    res.to_excel(output_file, index=True)

    # Ranger les metrics dans une dataframe que l'on retourne
    backtest_frame = pd.DataFrame({
        "Hit_ratio" : [round(best_hit, 5)],
        "Hurt frequence" : [best_hurst],
        "Size Matrix" : [best_size_mat],
        "nHorizon" : [best_horizon],
        "Sharpe B&H" : [res["Sharpe Ratio"].mean()],
        "Sharpe Strategy" : [res["Sharpe Ratio Strategy"].mean()],
        "MDD B&H" : [res["Max Drawdown"].min()],
        "MDD Strategy" : [res["Max Drawdown Strategy"].min()],
        "P&L B&H" : [res["P&L"].iloc[-1]],
        "P&L  Strategy" : [res["P&L Strategy"].iloc[-1]],
        "VaR 95% B&H" : [res["VaR 95%"].quantile(0.05)],
        "VaR 99% B&H" : [res["VaR 99%"].quantile(0.05)],
        "VaR 95% B&H Strategy" : [res["VaR 95% Strategy"].quantile(0.05)],
        "VaR 99% B&H Strategy" : [res["VaR 99% Strategy"].quantile(0.05)],

    }, index=[ticker])

    return backtest_frame
    


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    
    timeframe = "1min"
    ticker = "TTE"
    path_file = rf"Data\\Backtest\\{timeframe}\\{ticker}.xlsx"

    df_metrics = pd.read_excel(path_file, index_col=0)

    plot_compare_metrics(df_metrics, ticker, "Sharpe Ratio")

    # df_metrics = pd.DataFrame(None)

    # for files in os.listdir(rf"Data\\Bloomberg Original Data\\{timeframe}")[:18]:
    #     ticker = files.split(".xlsx")[0]
    #     print(ticker)
    #     df = pipeline_from_data_to_metrics(timeframe=timeframe, ticker=ticker)
    #     df_metrics = pd.concat([df_metrics, df])
    
    # df_metrics.to_excel(rf"Data\\Backtest\\{timeframe}\\metrics.xlsx", index=True)

    pass