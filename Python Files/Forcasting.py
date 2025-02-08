import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
import json
from Hurst import HurstDistribution
import os

class Forecast:

    def __init__(self,
                 hurstobj : HurstDistribution,
                 hurst_fq : str,
                 start_date : str,
                 generation : int,
                 horizon : int,
                 h = None):
        
        # Data 
        self.hurstobj = hurstobj
        self.data = hurstobj.data
        self.ticker = hurstobj.ticker

        # Metrics 
        self.S0_index = None
        self.S0 = self.get_S0(start_date)
        
        self.hurst_fq = hurst_fq
        self.H = self.get_Hurst(hurstobj) if h is None else h

        self.vol = self.get_vol()


        # Forcasting 
        self.start = start_date
        self.horizon = horizon + 1
        self.generation = generation
        self.indexes = self.get_index_from_S0_to_Horizon()
        self.paths = pd.DataFrame(index=self.indexes)
        # self.forcasting()


    def get_Hurst(self, hurstDistrib : HurstDistribution):
        return hurstDistrib.get_current_Hurst(self.S0_index, self.hurst_fq)

    def get_S0(self, start_date):
        
        s0 = self.data.loc[start_date]

        self.S0_index = s0.name
        
        return s0["Price"]
    
    def get_vol(self):
        return np.std(self.data["Log Return"]) * np.float_power(252, self.H)


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
        

    def forcasting(self, h_freq : str):
        """Pour l'horizon donné avec le nombre de génération.  
        Utilise une génération d'un fbm avec Cholesky.  
        Enregistre le forcast.

        Args:
            h_freq (str): Fréquence de H.
        """


        for i in range(self.generation):

            path = self.generate_fbm_cholesky()

            self.paths[i] = self.fbm_to_price(path)
            
        self.savePath(h_freq)

        # self.paths.to_parquet(f"\Data\Forecasting\{self.__str__()}.parquet")

        

    def get_index_from_S0_to_Horizon(self) -> list:

        df_after_s0 =  self.data.loc[self.S0_index:]

        max_len = len(df_after_s0)

        self.horizon = min(max_len, self.horizon)

        indexes = df_after_s0.iloc[:self.horizon].index.to_list()
        
        return indexes
    

    def plot_forcast(self, df : pd.DataFrame):

        plt.figure(figsize=(13,7))

        for c in df.columns:
            plt.plot(df[c], c=np.random.rand(3,))
        
        plt.title(self)
        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.grid(True, alpha = 0.7)
        plt.show()



    def savePath(self, h_freq : str):
        """Enregistre le forcast au format csv.

        Args:
            h_freq (str): Fréquence de Hurst pour ce forcast.
        """
        
        dir_path = f"Data\\Forecasting\\{h_freq}\\{self.horizon} Days"
        path_file = os.path.join(dir_path, f"{self}.csv")

        os.makedirs(dir_path, exist_ok=True)

        self.paths.to_csv(path_file, index=True)

        # self.read_forcast(path_file)

    def read_forcast(self, file_path : str) -> pd.DataFrame:
        """Lis et retourne le forecast en DataFrame.

        Args:
            file_path (str): chemin du fichier contenant le forcast

        Returns:
            pd.DataFrame: forcast
        """

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

        return df



    def save_forecast(self):
        """Sauvegarde self.paths au format json.
        """


        def convertir_ndarray(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError("Type non sérialisable")

        with open(f"\Data\Forecasting\{self.__str__}.json", "w") as file:
            json.dump(self.paths, file, indent=4, default=convertir_ndarray)

    def __str__(self):
        
        start_format = pd.to_datetime(self.start).strftime("%A %d %B %Y")
        end_format = pd.to_datetime(str(self.indexes[-1])).strftime("%A %d %B %Y")

        return f"{self.ticker} From {start_format} To {end_format} - Generations {self.generation} - Horizon {self.horizon}"