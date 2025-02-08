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

        

    def backtest(self):

        for date, h in self.H_distrib.items():

            forecast = Forecast(self.hurstobj,
                               self.hurst_fq,
                               date,
                               self.generation,
                               self.horizon,
                               h)
            
            #self.forecast = forecast
            
            forecast.forcasting(self.hurst_fq)

            self.original_data = pd.concat([self.original_data, self.get_compare_data(forecast)], axis=0)
            
            self.forecasted_data = pd.concat([self.forecasted_data,forecast.paths], axis=0)

            self.mean_forecasted_data = self.forecasted_data.mean(axis=1)

        self.hit_ratio = self.compute_Hit_ratio()




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