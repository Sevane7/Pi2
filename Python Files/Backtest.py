import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score, mean_squared_error
from Forcasting import Forecast
from Hurst import HurstDistribution
from typing import Dict
import os

class BacktestStrategy:

    def __init__(self,
                 hurstobj : HurstDistribution,
                 analyst_name : str,
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
        self.backtest()


    def __str__(self):
        return f"{self.hurstobj.ticker} - Hurst freq {self.hurst_fq} - Generations {self.generation} - Horizon {self.horizon}"


        
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
                # "6M": {1: 1000, 3: 1000, 5: 2000, 10: 3000, 25: 3000, 40: 5000, 85: 5500}
                "6M": {3: 1000},
                "3Y": {1: 1000},
                "5Y": {5: 2000}
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

        return self.data.loc[indexes, "Price"]
    

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

                for date, h in self.H_distrib.items():

                    forecast = Forecast(self.hurstobj,
                                    self.hurst_fq,
                                    date,
                                    self.generation,
                                    self.horizon,
                                    h)
                    
                    self.filling_data(forecast)

                # self.hit_ratio = self.compute_Hit_Ratio()

                self.mse = self.compute_MSE()

                self.save_metrics(forecast)

    def filling_data(self, forecast : Forecast) -> None:

        forecast.forcasting(self.hurst_fq)

        self.original_data = pd.concat([self.original_data, self.get_compare_data(forecast)], axis=0)
        
        self.forecasted_data = pd.concat([self.forecasted_data,forecast.paths], axis=0)

        self.mean_forecasted_data = self.forecasted_data.mean(axis=1)



    # Metrics methods
    def compute_MSE(self):

        dates_SO = self.H_distrib.keys()
        
        y_true = self.original_data.drop(dates_SO)

        y_pred = self.mean_forecasted_data.drop(dates_SO)

        # plt.plot(y_true, label="original data", c="blue")
        # plt.plot(y_pred, label ="forcast", c="red")
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        return mean_squared_error(y_true, y_pred)
    
    def compute_Hit_Ratio(self):

        dates_SO = self.H_distrib.keys()
        
        real_values = self.original_data.drop(dates_SO)
        
        if len(real_values) < self.horizon:
            print(f"Données réelles incomplètes pour comparaison. Comparaison sur {len(real_values)} jours seulement.")

        #Vérification directionnelle (hausse/baisse correcte)
        real_direction = np.sign(np.diff(real_values))  
        forecast_direction = np.sign(np.diff(self.forecasted_data[:self.horizon]))  

        hit_rate = (real_direction == forecast_direction).mean()  

        #RMSE
        rmse = np.sqrt(mean_squared_error(real_values[:self.horizon], 
                                        self.forecasted_data[:self.horizon]))

        # print(f"Ticker : {self.ticker}")
        # print(f"Taux de réussite directionnel : {hit_rate * 100:.2f}%")
        # print(f"Erreur RMSE : {rmse:.4f}")

        return hit_rate, rmse

    def save_metrics(self, forcast : Forecast):

        # Verifier si dossier et fichier existent    
        dir_path = f"Data\\Forecasting\\Metrics\\{self.hurstobj.dataobj.timeframe}"

        os.makedirs(dir_path, exist_ok=True)
        
        file_path = os.path.join(dir_path, f"{self.hurstobj.ticker}.xlsx")

        new_data = pd.DataFrame({
            "Forcast" : [forcast],
            # "Hit Ratio" : [self.hit_ratio],
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