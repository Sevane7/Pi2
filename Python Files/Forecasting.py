import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Hurst import HurstDistribution
from scipy.linalg import cholesky
from abc import abstractmethod

class Forecast:

    def __init__(self,
                 h_data : pd.DataFrame,
                 h_freq : str,
                 date : str,
                 horizon : int):
        
        # Data 
        self.h_data = h_data
        self.h_freq = h_freq

        # Metrics 
        self.S0_index = date
        self.s0 = self.get_S0()
        self.H = self.get_Hurst()
        self.vol = self.get_vol()


        # Forcasting 
        self.horizon = horizon + 1
        self.indexes = self.get_index_from_S0_to_Horizon()


    def __str__(self):
        
        start_format = pd.to_datetime(self.start).strftime("%A %d %B %Y")
        end_format = pd.to_datetime(str(self.indexes[-1])).strftime("%A %d %B %Y")

        return f"{self.ticker} From {start_format} To {end_format} - Generations {self.generation} - Horizon {self.horizon - 1}"


    # Get methods 
    def get_Hurst(self):
        return self.h_data.loc[self.S0_index, f"Hurst {self.h_freq}"]

    def get_S0(self):       
        return self.h_data.loc[self.S0_index, "Log Price"]
    
    def get_vol(self):
        return np.std(self.h_data["Log Price"]) * np.float_power(252, self.H)

    def get_index_from_S0_to_Horizon(self) -> list:

        df_after_s0 =  self.h_data.loc[self.S0_index:]

        self.horizon = min(len(df_after_s0), self.horizon)

        indexes = df_after_s0.iloc[:self.horizon].index.to_list() 
        # faire un [1:self.horizon] pour résoudre le pb de s0 dans le forecast
        
        return indexes


    @abstractmethod
    def forcasting(self):
        pass

    # Plot methods
    def plot_forcast(self, df : pd.DataFrame):

        plt.figure(figsize=(13,7))

        for c in df.columns:
            plt.plot(df[c], c=np.random.rand(3,))
        
        plt.title(self)
        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.grid(True, alpha = 0.7)
        plt.show()


class MonteCarlo(Forecast):

    def __init__(self,
                 h_data : pd.DataFrame,
                 h_freq : str,
                 date : str,
                 horizon : int,
                 genereation : int):
        
        super().__init__(h_data, h_freq, date, horizon)
        
        self.generation = genereation

        self.paths = pd.DataFrame(index=self.indexes)


    def generate_fbm_cholesky(self, epsilon = 1e-10) -> np.ndarray :
        """Generate a fractionnal Brownian Motion with cholesky method.

        Args:
            n (int): size of the process
            hurst (float): Hurst exponent of the process
            epsilon (type, optional): Defaults to 1e-10.

        Raises:
            ValueError: Not positive matrix

        Returns:
            np.ndarray: fBm process
        """

        # Definition of the time steps
        t = np.linspace(0, 1, self.horizon)

        # Creation of the covariance matrix
        cov_matrix = np.zeros((self.horizon, self.horizon))

        for i in range(self.horizon):
            for j in range(self.horizon):
                cov_matrix[i, j] = 0.5 * (t[i]**(2*self.H) + t[j]**(2*self.H) - np.abs(t[i] - t[j])**(2*self.H))

        cov_matrix += np.eye(self.horizon) * epsilon

        # Cholesky decomposition
        try:
            L = cholesky(cov_matrix, lower=True)

        except np.linalg.LinAlgError:

            raise ValueError("Cov Matrix is not positive definite.")

        # Generate random Gaussian variables
        z = np.random.normal(size=self.horizon)

        # Generate the fBm sample
        fbm_sample = np.array(L @ z)
        
        return fbm_sample

    def fbm_to_price(self, path : np.ndarray) -> np.ndarray:
        """
        Transforms a fractional Brownian motion sample into a price series
        starting at S0 with exponential (multiplicative) variations.

        fbm_sample: fBm values (starting around 0)
        S0        : desired initial price
        alpha     : volatility scale (or intensity of variations)
        """
        return self.S0 * np.exp(self.vol * (path - path[0]))   

    def forcasting(self):
        """Pour l'horizon donné avec le nombre de génération.  
        Utilise une génération d'un fbm avec Cholesky.  
        """

        for i in range(self.generation):

            path = self.generate_fbm_cholesky()

            self.paths[i] = self.fbm_to_price(path)



class CovarianceBased(Forecast):

    def __init__(self,
                 h_data : pd.DataFrame,
                 h_freq : str,
                 date : str,
                 horizon : int
                 ):
        
        super().__init__(h_data.loc[:date], h_freq, date, horizon)

        self.t_n = len(self.h_data.loc[:date])

        self.sigma_Y = self.get_sigma_Y()

        self.sigma_XY = self.get_sigma_XY()


    def get_cov_fbm(self, sigma : float, s : int, t: int, hurst : float) -> float: 
        return (
            (pow(sigma, 2) / 2) * (
            np.abs(s)**(2*hurst) + np.abs(t)**(2*hurst) - np.abs(s-t)**(2*hurst))
            )
    
    def get_sigma_Y(self) -> np.ndarray:

        res = np.ndarray((self.t_n, self.t_n))

        for i in range(1, res.shape[0]+1):
            for j in range(1, res.shape[1]+1):
                res[i-1,j-1] = self.get_cov_fbm(self.vol, i, j, self.H)

        return res

        
    
    def get_sigma_XY(self) -> np.ndarray:

        t_plus_h = self.t_n + self.horizon

        return np.array([self.get_cov_fbm(self.vol, t_plus_h, i, self.H) for i in range(1, self.t_n + 1)])


    def get_sigma_X(self, t= 0) -> float:
        return (pow(self.vol, 2) / 2) * (2 * np.abs(t + self.horizon)**(2*self.H)) 


    def forecasting(self) -> float:
        
        xy = self.sigma_XY.reshape(-1,1)
        Y = self.h_data["Log Price"].to_numpy().reshape(-1,1)

        X_t_h_Y = xy.T @ np.linalg.inv(self.sigma_Y) @ Y

        return X_t_h_Y.item()
    