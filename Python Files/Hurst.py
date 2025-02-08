import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from BloombergData import BloombergData


class HurstDistribution:

    def __init__(self,
                 originale_data : BloombergData):
        
        self.dataobj : BloombergData = originale_data
        
        self.data : pd.DataFrame = originale_data.bounded_data
        
        self.frequencies = ["1W", "2W","1M", "3M", "6M", "1Y", "3Y", "5Y"] # adapter changer pour intraday

        self.ticker : str = originale_data.ticker

        self.data = self.hurst_distribution(originale_data.bounded_data)
          

    def logPeriodo(self, fbm, m1=None, m2=None) -> float:
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
            return None

    def hurst_distribution(self, df : pd.DataFrame) -> pd.DataFrame:

        for fq in self.frequencies:
            
            grouped_values = df.groupby(pd.Grouper(freq=fq))["Price"]
            hurst_values = grouped_values.apply(self.logPeriodo)
            
            df[f"H_{fq}"] = hurst_values.reindex(df.index, method="ffill")
        
        return df

    def get_current_Hurst(self, date : str, freq_hurst) -> float:

        return self.data.loc[date, f"H_{freq_hurst}"]

    def get_changes_Hurst(self, freq_hurst : str) -> pd.Series:
        """Retourne la Series à la date où les Hurst sont calculés

        Args:
            freq_hurst (str): Fréquence de distribution de Hurst

        Returns:
            pd.Series: Distribution espacée par la fréquence de Hurst
        """

        return self.data.loc[self.data[f"H_{freq_hurst}"] != self.data[f"H_{freq_hurst}"].shift(), f"H_{freq_hurst}"]



    def plot_distrib(self, freq : str):

        plt.figure(figsize=(12,8))

        plt.plot(self.data[f"H_{freq}"])

        plt.title(f"H {self.originale_data} - Window {freq}")

        plt.xlabel("Date")

        plt.ylabel("Hurst Exponent")
        
        plt.grid(True, alpha=0.7)  
        
        plt.show() 