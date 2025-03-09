import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from Forecasting import load_forecast
from Metrics import *
from typing import Union

class BacktestStrategy:

    def __init__(self,
                 ticker : str,
                 timeframe : str,
                 h_freq : str,
                 horizon : int,
                 forecast : Union[pd.DataFrame, pd.Series],
                 ):
        
        # First attributs
        self.ticker = ticker
        self.timeframe = timeframe
        self.h_freq = h_freq
        self.horizon = horizon

        if isinstance(forecast, pd.Series):
            self.data : pd.DataFrame = self.get_data(forecast)
        elif isinstance(forecast, pd.DataFrame):
            self.data = forecast


        # Metrics
        self.hit_ratio : float = self.compute_Hit_Ratio()
        self.mse : float = self.compute_MSE()


    def __str__(self):
        return f"{self.ticker} - {self.timeframe} - Hurst Freq {self.h_freq} - Horizon {self.horizon}"


    def get_data(self, forecast : pd.Series) -> pd.DataFrame:

        file = rf"Data\\Data Hurst - Final\\{self.ticker}.xlsx"

        indexes = forecast.index

        data = pd.read_excel(file, index_col=0, sheet_name=self.timeframe)

        data = data.loc[indexes, "Log Price"]

        data = pd.concat([data, forecast.rename("Forecasted Price")], axis = 1)

        return data.dropna()

    
    def compute_MSE(self):
        
        y_true = self.data["Log Price"].to_list()

        y_pred = self.data["Forecast"].to_list()

        return mean_squared_error(y_true, y_pred)
    
    def compute_Hit_Ratio(self):

        y_true = self.data["Log Price"].to_numpy()

        y_pred = self.data["Forecast"].to_numpy()
        
        real_direction = np.sign(np.diff(y_true.flatten()))  
        forecast_direction = np.sign(np.diff(y_pred.flatten()))  

        min_len = min(len(real_direction), len(forecast_direction))
        real_direction = real_direction[:min_len]
        forecast_direction = forecast_direction[:min_len]

        res = (real_direction == forecast_direction)

        hit_rate = res.mean()

        return hit_rate

    def save_metrics(self, size_matrix : str):

        # Verifier si dossier et fichier existent    
        file_path = r"Data\\Forecasting\\Metrics\\Covariance Based Hit ratio and MSE.xlsx"

        new_data = pd.DataFrame({
            "Size Matrix" : [size_matrix],
            "Asset" : [self.ticker],
            "Timeframe" : [self.timeframe],
            "Hurst Frequence" : [self.h_freq],
            "Horizon" : [self.horizon],
            "From" : [self.data.index[0]],
            "To" : [self.data.index[-1]],
            "Hit Ratio" : [self.hit_ratio],
            "MSE" : [self.mse]
        })

        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df = pd.concat([df, new_data], ignore_index=True)
        
        else:
            df = new_data

        df.to_excel(file_path, index=False)


    # Plotting methods
    def plot_original_VS_forcast(self):
        
        plt.figure(figsize=(12,8))
        plt.plot(self.data["Forecasted Price"], label="Forecasted Price", color="blue")
        plt.plot(self.data["Log Price"], label = "Real Price", color="red")
        plt.title(self)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.7)  
        plt.show()



def process_backtest(dir_forecast : str) -> None:
    """Effectue un backtest de la stratégie en calculant le Hit Ratio et MSE sur l'ensemble des forecast.  
    Enregistre dans un excel au chemin "Data\\Forecasting\\Metrics\\Covariance Based.xlsx"

    Args:
        dir_forecast (str): chemin du dictionnaire contenant les forecast.
    """

    files = [f for f in os.listdir(dir_forecast) if f.endswith(".json")]
    timeframe = "Daily"

    metrics : pd.DataFrame = None

    for file in files:
        print(file)

        ticker = file.split(".json")[0]


        file_path = os.path.join(dir_forecast, file)

        dic_forecast = load_forecast(file_path)

        
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

                        new_data = pd.DataFrame({
                            "Size Matrix" : [size_mat],
                            "Asset" : [ticker],
                            "Timeframe" : [timeframe],
                            "Hurst Frequence" : [h_fq],
                            "Horizon" : [horizon],
                            "From" : [backtestobj.data.index[0]],
                            "To" : [backtestobj.data.index[-1]],
                            "Hit Ratio" : [backtestobj.hit_ratio],
                            "MSE" : [backtestobj.mse]
                        })

                        metrics = pd.concat([new_data, metrics], axis = 0)

                    except Exception as e:
                        print(e)
                        print(size_mat, h_fq, horizon)

        with pd.ExcelWriter(r"Data\\Forecasting\\Metrics\\Covariance Based.xlsx") as xls:
        
            metrics.to_excel(xls, index=False)
                    

def get_accurate_asset(file_path : str) -> pd.DataFrame:

    df_metrics = pd.read_excel(file_path)

    accurate_assets : pd.DataFrame = None

    assets = df_metrics["Asset"].value_counts().index.to_list()

    for asset in assets:
        df_asset : pd.DataFrame = df_metrics[df_metrics["Asset"] == asset]

        best_asset = df_asset.sort_values(["Hit Ratio"], ascending=False).iloc[0]

        if(best_asset["Hit Ratio"] >= 0.5):
            accurate_assets = pd.concat([accurate_assets, best_asset.to_frame().T], ignore_index=True)

    return accurate_assets


def join_forecast(row : pd.Series, dir_dico : str, dir_data : str):


    path_data = os.path.join(dir_data, row["Asset"]  + ".xlsx")
    path_dico = os.path.join(dir_dico, row["Asset"]  + ".json")

    data : pd.DataFrame = pd.read_excel(path_data, sheet_name=row["Timeframe"], index_col=0)
    dico : dict[int, dict[str, dict[int, pd.DataFrame]]] = load_forecast(path_dico)


    df_forecast = dico[str(row["Size Matrix"])][row["Hurst Frequence"]][str(row["Horizon"])]
    
    res = data.merge(df_forecast["Forecast"], left_index=True, right_index=True, how="inner")

    hurst_cols = [c for c in data.columns if c.startswith("Hurst")]
    hurst_cols.remove(f"Hurst {row['Hurst Frequence']}")

    return res.drop(columns=hurst_cols) 



if __name__ == "__main__":

    dir_forecast = r"Data\\Forecasting\\Covariance Based\\Single Forecast\\Daily - Copie"

    def asset_to_compute_metrix():

        dir_data = r"Data\\Final Data - Copie"
        file_metrics = r"Data\\Forecasting\\Metrics\\Covariance Based.xlsx"


        # process_backtest(dir_forecast)

        df : pd.DataFrame = get_accurate_asset(file_metrics)

        for i in df.index:

            row = df.loc[i]

            print(row["Asset"])

            df_with_forecast = join_forecast(row=row, dir_data=dir_data, dir_dico=dir_forecast)

            df_with_forecast.to_excel(rf"Data\\Accurate Files\\{row["Asset"]}.xlsx", index=True)

    # process_backtest(dir_forecast=dir_forecast)

    # asset_to_compute_metrix()

    pass