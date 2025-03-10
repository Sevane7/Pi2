import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.optimize import minimize
from scipy.special import gamma
from BloombergData import BloombergData
from Entropy import symbolize_market_info
import warnings

class HurstDistribution(BloombergData):

    def __init__(self,
                 ticker : str,
                 timeframe : str,
                 start_date = None,
                 end_date = None):
        
        super().__init__(ticker, timeframe, start_date, end_date)

        # Fréquences de Hurst en fonction de la timeframe
        self.frequencies = list()
        self.freq_window = dict[str:int]
        self.fill_freq()

        # 2 Datafrmes :
            # Une avec toutes les données
            # Une avec les données à backtest
        self.data = self.load_data()

        self.prepared_data = self.hurst_distribution()

    # Construcuteur
    def convert_to_days(self, fq : str) -> int:

        n_period = int(fq[0])

        if fq.endswith("W"):
            return n_period * 5
        
        elif fq.endswith("M"):
            return n_period * 21
        
        elif fq.endswith("Y"):
            return n_period * 252

    def fill_freq(self) -> None:
    
        if(self.timeframe == "Daily"):          
            self.frequencies = ["2W","1M", "3M", "6M", "1Y", "3Y", "5Y"]

            self.freq_window = {
                fq : self.convert_to_days(fq) for fq in self.frequencies
            }
        
        else:
            self.frequencies = ["15min", "30min", "1h", "3h", "6h", "12h"] 

            self.freq_window = {
                fq : int(pd.Timedelta(fq).total_seconds() // 60) for fq in self.frequencies
            }    

    # Whittle Likelyhood
    def spdFGN(self, Htry : float, n : int) -> np.ndarray:
        """Calcule la densité spectrale normalisée du fGn à des fréquences de Fourier.

        Args:
            Htry (float): Valeur du paramètre H pour lequel calculer la densité spectrale.
            n (int): Longueur de la série temporelle.

        Returns:
            np.ndarray: Densité spectrale.
        """
        
        alpha = 2 * Htry + 1
        
        nstar = (n - 1) // 2
        
        clambda = (np.sin(np.pi * Htry) * gamma(alpha)) / np.pi

        spd = np.zeros(nstar)

        for k in range(1, nstar + 1):
            lambda_k = (2 * np.pi * k) / n
            sum_term = np.sum(np.abs(2 * np.pi * np.arange(-300, 301) + lambda_k) ** (-alpha))
            spd[k - 1] = clambda * (1 - np.cos(lambda_k)) * sum_term

        # Renormalisation de spd
        theta = np.exp((2 / n) * np.sum(np.log(spd)))
        
        spd /= theta

        return spd

    def whittle_likelihood(self, Htry : float, perio : np.ndarray, n : int) -> float:
        """Fonction de vraisemblance Whittle à minimiser.

        Args:
            Htry (float): Valeur du paramètre H.
            perio (np.ndarray):  Périodogramme de la série temporelle.
            n (int): Longueur de la série temporelle.

        Returns:
            float: Estimateur de wittle par vraissenblance.
        """
        
        spd = self.spdFGN(Htry, n)

        return (4 * np.pi / n) * np.sum(perio / spd)

    def whittle_estimator(self, fBm : np.ndarray) -> float:
        """Estimateur de Whittle pour le paramètre H.

        Args:
            fBm (np.ndarray): érie temporelle simulée.
            Hprel (float, optional): Estimation initiale de H.. Defaults to 0.5.

        Returns:
            float: _description_
        """
        
        # Convertir le fBm en fGn
        fGn = np.diff(fBm, prepend=0)
        n = len(fGn)
        nstar = (n - 1) // 2

        # Calcul du périodogramme
        perio = (np.abs(fft(fGn)) ** 2) / (2 * np.pi * n)
        perio = perio[:nstar]

        # Minimisation de la fonction de vraisemblance Whittle
        result = minimize(self.whittle_likelihood,
                        x0=0.5,
                        args=(perio, n),
                        bounds=[(0.01, 0.99)])

        H = result.x[0]

        return H

    # Log Periodogram   
    def logPeriodo(self, fbm : np.ndarray, m1=None, m2=None) -> float:
        """
        Estimation du paramètre H en utilisant le log-périodogramme.

        m1, m2 : Plages de fréquences utilisées.
        llplot : Si True, réalise un plot log-log des fréquences et du spectre.

        Retourne : Estimation du paramètre H.
        """

        # Convertir le fBm en fGn (différences successives)
        fGn = np.diff(fbm, prepend=0)

        # Par défaut, plages des fréquences
        if m1 is None:
            m1 = 1
        if m2 is None:
            m2 = len(fbm) // 2


        # Calcul du périodogramme et des fréquences de Fourier
        I_lambda = (np.abs(fft(fGn))**2) / (2 * np.pi * len(self.data))
        I_lambda = I_lambda[m1:m2]
        lambda_vals = (2 * np.pi * np.arange(m1, m2)) / len(self.data)

        # Régression linéaire log-log
        log_lambda = np.log(lambda_vals)
        log_I_lambda = np.log(I_lambda)

        try:
            slope, intercept = np.polyfit(log_lambda, log_I_lambda, 1)
            return 0.5 * (1 - slope)
        except:
            return np.nan

    # Apply Hurst Methods
    def hurst_distribution(self) -> pd.DataFrame:

        for fq, window in self.freq_window.items():

            print(fq)

            if fq == "1W" or fq == "5min":
                self.data[f"Hurst {fq}"] = (
                    self.data["Log Price"]
                    .rolling(window=window,
                            min_periods=2)
                    .apply(self.whittle_estimator,
                        raw=True)
                    )
                
            else:
                self.data[f"Hurst {fq}"] = (
                    self.data["Log Price"]
                    .rolling(window=window,
                            min_periods=2)
                    .apply(self.logPeriodo,
                        raw=True)
                    )                
        
        return self.bound_data(self.data)



def plot_distrib(dir_path : str, ticker : str, freq : list, timeframe : str, start_index = None, end_index = None):

    file_path = os.path.join(dir_path, f"{ticker}.xlsx")

    distrib : pd.DataFrame = pd.read_excel(file_path, sheet_name=timeframe ,index_col=0)

    start_index = 0 if start_index == None else start_index
    end_index = len(distrib) if end_index == None else end_index

    distrib = distrib.iloc[start_index : end_index]

    hurst_columns : list = [f"Hurst {fq}" for fq in freq if f"Hurst {fq}" in distrib.columns]

    plt.figure(figsize=(13,7))

    for i, col in enumerate(hurst_columns):

        plt.plot(distrib[col], label = col, color = np.random.rand(3,))

    plt.title(f"Hurst Distribution - {timeframe} {ticker}")

    plt.xlabel("Date")

    plt.ylabel("Hurst Exponent")

    plt.legend()
    
    plt.grid(True, alpha=0.7)  
    
    plt.show() 


def apply_hurst(ticker, timeframe, min_drop = 362, daily_drop = 758) -> pd.DataFrame:
            
    df = HurstDistribution(ticker=ticker, timeframe=timeframe).prepared_data

    drop_lign = min_drop if timeframe == "1min" else daily_drop

    df = df.iloc[drop_lign:]

    return df
    



if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # daily_frequencies = ["2W", "1M", "3M", "6M", "1Y", "3Y", "5Y"]
    # minutes_frequencies = ["15min", "30min", "1h", "3h", "6h", "12h"]

    daily = "Daily"
    minutes = "1min"
    ticker = "BTC"

    apply_hurst(ticker=ticker, timeframe=daily)

    # plot_distrib(dir_path, ticker, minutes_frequencies, minutes)
    
    pass
