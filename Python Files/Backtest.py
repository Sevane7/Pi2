import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from Forecasting import load_forecast
from Entropy import symbolize_market_info

class BacktestStrategy:

    def __init__(self,
                 ticker : str,
                 timeframe : str,
                 h_freq : str,
                 horizon : int,
                 forecast : pd.Series,
                 ):
        
        # First attributs
        self.ticker = ticker
        self.timeframe = timeframe
        self.h_freq = h_freq
        self.horizon = horizon

        self.data : pd.DataFrame = self.get_data(forecast)   

        # Metrics
        self.hit_ratio : float =self.compute_Hit_Ratio()
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

        y_pred = self.data["Forecasted Price"].to_list()

        return mean_squared_error(y_true, y_pred)
    
    def compute_Hit_Ratio(self):

        y_true = self.data["Log Price"].to_numpy()

        y_pred = self.data["Forecasted Price"].to_numpy()
        
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
        file_path = f"Data\\Forecasting\\Metrics\\Covariance Based Hit ratio and MSE.xlsx"

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

    def save(self):

        dirpath = f"Data\\Forecasting\\{self.hurst_fq}"

        os.makedirs(dirpath, exist_ok=True)


        file_path = os.path.join(dirpath, f"{self}.xlsx")

        # self.mean_forecasted_data.to_excel(file_path, sheet_name="Forcast", index=True)
        # self.original_data.to_excel(file_path, sheet_name='Original', index=True)

        # final_orignal = pd.concat([original_data, original_returns], axis=1)
        # final_forecasted = pd.concat([forecasted_data, forecasted_returns], axis=1)
        # final_orignal = final_orignal[['Price', 'Returns', 'Efficiency_Indicator']]
        # final_forecasted = final_forecasted[['Price', 'Returns', 'Efficiency_Indicator']]

        # # Variante 1 de l'écriture (2 feuilles dans meme fichier)
        with pd.ExcelWriter(file_path) as writer:
            self.original_data[["Price", "Log Return", "Efficiency Indicator"]].to_excel(writer, sheet_name='Original', index=True)
            self.mean_forecasted_data.to_excel(writer, sheet_name='Forecast', index=True)


    """
        # Running backtest
    def backtest(self):
        Forcast pour toute la période donnée.  
        Initialise les attributs hurst_fq, horizon, generation, et les data (originale, forcast, mean_forcast).  
        

        for h_freq, horizon_gene in self.combi_generations.items():

            self.hurst_fq = h_freq

            for horiz, gene in horizon_gene.items():

                self.horizon = horiz

                self.generation = gene
                
                self.H_distrib = self.hurstobj.get_changes_Hurst(self.hurst_fq).dropna()

                print(datetime.now().strftime("%H:%M:%S"))
                print(f"Running {self}")
                
                for date, h in self.H_distrib.items():

                    forecast = Forecast(self.hurstobj,
                                    self.hurst_fq,
                                    date,
                                    self.generation,
                                    self.horizon,
                                    h)
                    
                    self.filling_data(forecast)
                               
                self.compute_mean_forecated_data()
                
                self.save()
                
                self.hit_ratio = self.compute_Hit_Ratio()

                self.mse = self.compute_MSE()

                self.save_metrics(forecast)

    def simple_backtest(self, h_fq, N_generations, horizons):
        
        self.hurst_fq = h_fq
        self.H_distrib = self.hurstobj.get_changes_Hurst(self.hurst_fq).dropna()
        self.horizon = horizons
        self.generation = N_generations
                
        for date, h in self.H_distrib.items():

            forecast = Forecast(self.hurstobj,
                            self.hurst_fq,
                            date,
                            self.generation,
                            self.horizon,
                            h)
                    
            self.filling_data(forecast)
                                
        self.compute_mean_forecated_data()

        self.save()
        
        # self.hit_ratio = self.compute_Hit_Ratio()

        # self.mse = self.compute_MSE()

    def filling_data(self, forecast) -> None:

        forecast.forcasting(self.hurst_fq)

        self.original_data = pd.concat([self.original_data, self.get_compare_data(forecast)], axis=0)
        
        self.forecasted_data = pd.concat([self.forecasted_data,forecast.paths], axis=0)
    
    def compute_mean_forecated_data(self):

        self.mean_forecasted_data = pd.DataFrame(self.forecasted_data.mean(axis=1), columns=["Price"])

        self.mean_forecasted_data["Log Return"] = np.log(self.mean_forecasted_data["Price"] / self.mean_forecasted_data["Price"].shift())
        
        # self.mean_forecasted_data["Log Return"].loc[self.mean_forecasted_data["Log Return"].reset_index().index % self.horizon == 0] = np.nan

        # compute_entropy_indicator(self.mean_forecasted_data, 5, 2)

    """

def save_hit_mse_strategy():

    for file in os.listdir(f"Data\\Forecasting\\Covariance Based"):

        name_file = file.split(".json")[0]
        size_mat = name_file.split(" ")[-1]

        print(file)

        loaded_f = load_forecast(name_file)

        for tick in loaded_f.keys():
            # print(tick, end="\n\t")

            for timeframe in loaded_f[tick].keys():
                # print(timeframe, end="\n\t\t")

                for h_freq in loaded_f[tick][timeframe].keys():
                    # print(h_freq, end="\n\t\t\t")

                    for horizon, forecast in loaded_f[tick][timeframe][h_freq].items():

                        current_backtest = BacktestStrategy(ticker=tick,
                                                            timeframe=timeframe,
                                                            h_freq=h_freq,
                                                            horizon=horizon,
                                                            forecast=forecast)
                        
                        current_backtest.save_metrics(size_mat)

def calculate_vol(series : pd.Series, window : int) :
    return series.rolling(window).std() * np.sqrt(252)

def calculate_sharpe(df : pd.DataFrame, risk_free_rate=0):
    rolling_mean = df["Return"]
    rolling_std = df["Volatility"]
    sharpe_ratio = (rolling_mean - risk_free_rate) / rolling_std
    return sharpe_ratio

def calculate_maxDrawdown(series : pd.Series, window : int):
    rolling_max = series.rolling(window=window).max()
    drawdown = series / rolling_max - 1
    max_drawdown = drawdown.rolling(window=window).min()
    return max_drawdown

def calculate_Cummul_Return(log_returns : pd.Series):
    return (1 + log_returns).cumprod() - 1
        

def process_metrics():
    
    folder_path = r"Data\\Data Hurst - Final - Copie"

    excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    for file in excel_files:
        print(file)
        file_path = os.path.join(folder_path, file)
        
        # Charger toutes les feuilles
        xls = pd.ExcelFile(file_path)
        sheets : dict[str, pd.DataFrame]= {}
        
        for sheet_name in xls.sheet_names:
            print(sheet_name)
                  
            df = xls.parse(sheet_name)

            # Ici ajouter les methodes sortino, VaR
            df["Price"] = np.exp(df["Log Price"])
            df["Return"] = df["Price"].diff() / df["Price"]
            df["Log Return"] = df["Log Price"].diff()
            df["Volatility"] = calculate_vol(df["Log Return"], 252)
            df["MaxDrawdown"] = calculate_maxDrawdown(df["Price"], 252)
            df["Sharpe"] = calculate_sharpe(df)

            sheets[sheet_name] = df.dropna()
        
        # Sauvegarder en écrasant
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)





if __name__ == "__main__":

    pass