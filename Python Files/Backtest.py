import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score, mean_squared_error
from Forcasting import Forecast
from Hurst import HurstDistribution
from typing import Dict
import os
from Entropy import compute_entropy_indicator
from datetime import datetime

class BacktestStrategy:

    def __init__(self,
                 hurstobj : HurstDistribution,
                 analyst_name : str,
                 simple_backtest : bool
                 ):
        
        # First attributs
        self.hurstobj : HurstDistribution = hurstobj
        self.data : pd.DataFrame = hurstobj.data        
        self.combi_generations = self.get_analyst_combinations(analyst_name)

        # Hurst
        self.hurst_fq : str
        self.H_distrib : pd.Series

        # Forcast
        self.generation : int
        self.horizon : int

        # Forcast Data
        self.original_data = pd.DataFrame()
        self.forecasted_data = pd.DataFrame()
        self.mean_forecasted_data = pd.DataFrame()

        # Metrics
        self.hit_ratio : float
        self.mse : float

        # Running Backtest
        if simple_backtest == False:
            self.backtest()


    def __str__(self):
        return f"{self.hurstobj.dataobj} - Hurst freq {self.hurst_fq} - Generations {self.generation} - Horizon {self.horizon}"


        
    # Get methods 
    def get_analyst_combinations(self, analyst_name : str) -> dict[str, dict[int, int]] : 
        data : dict[str, dict[int, int]] = {
            "Ghali": {
                "1M": {1: 1000, 3: 1000, 5: 2000, 7: 3000, 10: 3000, 20: 5000},
                "5Y": {1: 1000, 3: 1000, 5: 2000, 10: 3000, 25: 3000, 40: 5000, 85: 5500, 120: 6400, 200: 7300, 300: 8200, 400: 9100, 500: 10000, 650: 10900, 900: 20000}
            },
            "Jules": {
                "2W": {1: 1000, 3: 1000, 5: 2000, 7: 3000, 10: 3000},
                "3M": {1: 1000, 3: 1000, 5: 2000, 10: 3000, 25: 3000, 40: 5000}
            },
            "Julie": {
                "6M": {1: 1000, 3: 1000, 5: 2000, 10: 3000, 25: 3000, 40: 5000, 85: 5500}
            },
            "Sevane": {
                "1Y": {1: 1000, 3: 1000, 5: 2000, 10: 3000, 25: 3000, 40: 5000, 85: 5500, 120: 6400, 200: 7300},
                "3Y": {1: 1000, 3: 1000, 5: 2000, 10: 3000, 25: 3000, 40: 5000, 85: 5500, 120: 6400, 200: 7300, 300: 8200, 400: 9100, 500: 10000}
            },
            "Tommy": {
                "1W": {1: 1000, 3: 1000, 5: 2000}
            }
        }

        return data[analyst_name]

    def get_compare_data(self, forecast : Forecast) -> pd.DataFrame:

        indexes = forecast.get_index_from_S0_to_Horizon()

        return self.data.loc[indexes]
    

    # Running backtest
    def backtest(self):
        """Forcast pour toute la période donnée.  
        Initialise les attributs hurst_fq, horizon, generation, et les data (originale, forcast, mean_forcast).  
        """

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


    def filling_data(self, forecast : Forecast) -> None:

        forecast.forcasting(self.hurst_fq)

        self.original_data = pd.concat([self.original_data, self.get_compare_data(forecast)], axis=0)
        
        self.forecasted_data = pd.concat([self.forecasted_data,forecast.paths], axis=0)
    
    def compute_mean_forecated_data(self):

        self.mean_forecasted_data = pd.DataFrame(self.forecasted_data.mean(axis=1), columns=["Price"])

        self.mean_forecasted_data["Log Return"] = np.log(self.mean_forecasted_data["Price"] / self.mean_forecasted_data["Price"].shift())
        
        # self.mean_forecasted_data["Log Return"].loc[self.mean_forecasted_data["Log Return"].reset_index().index % self.horizon == 0] = np.nan

        # compute_entropy_indicator(self.mean_forecasted_data, 5, 2)





    # Metrics methods
    def compute_MSE(self):

        dates_SO = self.H_distrib.keys()
        
        y_true = self.original_data.loc[:, "Price"].drop(dates_SO)

        y_pred = self.mean_forecasted_data.loc[:, "Price"].drop(dates_SO)

        # plt.plot(y_true, label="original data", c="blue")
        # plt.plot(y_pred, label ="forcast", c="red")
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        return mean_squared_error(y_true, y_pred)
    
    def compute_Hit_Ratio(self):
        dates_SO = self.H_distrib.keys()
    
        real_values = self.original_data.drop(dates_SO)
        forecast_values = self.mean_forecasted_data.drop(dates_SO)  
        
        if len(real_values) < self.horizon:
            print(f"Données réelles incomplètes pour comparaison. Comparaison sur {len(real_values)} jours seulement.")

        real_direction = np.sign(np.diff(real_values.to_numpy().flatten()))  
        forecast_direction = np.sign(np.diff(forecast_values.to_numpy().flatten()))  

        min_len = min(len(real_direction), len(forecast_direction))
        real_direction = real_direction[:min_len]
        forecast_direction = forecast_direction[:min_len]

        hit_rate = (real_direction == forecast_direction).mean()

        return hit_rate

    def save_metrics(self, forcast : Forecast):

        # Verifier si dossier et fichier existent    
        dir_path = f"Data\\Forecasting\\Metrics\\{self.hurstobj.dataobj.timeframe}"

        os.makedirs(dir_path, exist_ok=True)
        
        file_path = os.path.join(dir_path, f"{self.hurstobj.ticker}.xlsx")

        new_data = pd.DataFrame({
            "Forcast" : [self],
            "From" : [self.hurstobj.dataobj.start],
            "To" : [self.hurstobj.dataobj.end],
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
    def plot_original_VS_forcast(self, scatter = True):
        
        if scatter: 
            plt.scatter(self.original_data.index, self.original_data, c='red', label='original_data', marker='o', s=10, alpha=0.7)
            plt.scatter(self.mean_forecasted_data.index, self.mean_forecasted_data, c='blue', label='mean_forecasted_data', marker='s', s=10, alpha=0.7)
        
        else:
            plt.plot(self.original_data, c='red', label='original_data', alpha=0.7)
            plt.plot(self.mean_forecasted_data, c='blue', label='mean_forecasted_data', alpha=0.7)
        
        plt.title('Comparaison data réel et forecasted')
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.ylabel("Price")
        plt.legend()

        plt.grid(True, alpha = 0.7)
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
