import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score, mean_squared_error
from Forcasting import Forecast
from Hurst import HurstDistribution

class BacktestStrategy:

    def __init__(self,
                 hurstobj : HurstDistribution,
                 hurst_fq : str,
                 generation : int,
                 horizon : int
                 ):
        
        # Data
        self.data : pd.DataFrame = hurstobj.data
        # self.initial_path = self.get_compare_data()

        # Hurst
        self.hurstobj : HurstDistribution = hurstobj
        self.hurst_fq : str = hurst_fq
        self.H_distrib : pd.Series = self.hurstobj.get_changes_Hurst(self.hurst_fq).dropna()

        # Forcast
        self.generation : int = generation
        self.horizon : int = horizon
        # self.forecasted_path = self.meanPath(str(self.forcaster))

        #self.forecast = None
        self.original_data = pd.DataFrame()
        self.forecasted_data = pd.DataFrame()
        self.mean_forecasted_data = pd.DataFrame()

        # Others
        self.hit_ratio : float
        self.mse : float

        

    def backtest(self):

        for date, h in self.H_distrib.items():

            forecast = Forecast(self.hurstobj,
                               self.hurst_fq,
                               date,
                               self.generation,
                               self.horizon,
                               h)
                        
            forecast.forcasting(self.hurst_fq)

            self.original_data = pd.concat([self.original_data, self.get_compare_data(forecast)], axis=0)
            
            self.forecasted_data = pd.concat([self.forecasted_data,forecast.paths], axis=0)

            self.mean_forecasted_data = self.forecasted_data.mean(axis=1)

        self.mse = self.compute_Hit_ratio()


    def analyst_generations(self, forcast_analyst : str) -> dict : 
        data = {
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

        return data[forcast_analyst]



    def get_compare_data(self, forecast : Forecast) -> pd.DataFrame:

        indexes = forecast.get_index_from_S0_to_Horizon()

        return self.data.loc[indexes, "Price"]
    
    def meanPath(path : str):

        forcasted_path = pd.read_csv(path, index_col=0)
        print(forcasted_path)

        forcasted_path.index = pd.to_datetime(forcasted_path.index)

        return forcasted_path.mean(axis=0)
    

    def compute_Hit_ratio(self):

        dates_SO = self.H_distrib.keys()
        
        y_true = self.original_data.drop(dates_SO)

        y_pred = self.mean_forecasted_data.drop(dates_SO)

        plt.plot(y_true, label="original data", c="blue")
        plt.plot(y_pred, label ="forcast", c="red")
        plt.grid(True)
        plt.legend()
        plt.show()

        return mean_squared_error(y_true, y_pred)
    

    def hit_function(self):

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


    def comparaison(self):
        
        plt.scatter(self.original_data.index, self.original_data, c='red', label='original_data', marker='o', s=10, alpha=0.7 )
        plt.scatter(self.mean_forecasted_data.index, self.mean_forecasted_data, c='blue', label='mean_forecasted_data', marker='s', s=10, alpha=0.7 )
        #plt.plot(self.original_data, c='red', label='original_data')
        #plt.plot(self.mean_forecasted_data, c='blue', label='mean_forecasted_data')
        
        plt.title('Comparaison data réel et forecasted')
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.ylabel("Price")
        plt.legend()

        plt.grid(True, alpha = 0.7)
        plt.show()