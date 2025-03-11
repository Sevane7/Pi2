import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from abc import abstractmethod
import json

class Forecast:
    """
    Objet permettant d'effectuer un forecast avec la distribution de l'exposant de Hurst sur les Log Price d'un actif. 
    """

    def __init__(self,
                h_data : pd.DataFrame,
                h_freq : str,
                date : str,
                horizon : int):
        
        """
        """
        # Data 
        self.h_data = h_data
        self.h_freq = h_freq

        # Metrics 
        self.S0_index = date
        self.s0 = self.get_S0()
        self.H = self.get_Hurst()
        self.vol = self.get_vol()


        # Forcasting 
        self.horizon = horizon
        self.indexes = self.get_index_from_S0_to_Horizon()

    # Get methods 
    def get_Hurst(self) -> float:
        """Retourne le Hurst de notre actif correspondant à la date et à la fréquence donnée.

        Returns:
            float: Hurst Exponent.
        """
        return self.h_data.loc[self.S0_index, f"Hurst {self.h_freq}"]

    def get_S0(self) -> float:
        """
        Retourne le Log Price initial au moment de notre forecast.

        Returns:
            float: Log Price
        """ 
        return self.h_data.loc[self.S0_index, "Log Price"]
    
    def get_vol(self) -> float:
        """Retourne la volatilité historique des Log Price annualisé avec le Hurst Exponent.

        Returns:
            float: Volatilité.
        """
        return np.std(self.h_data["Log Price"]) * np.float_power(252, self.H)

    def get_index_from_S0_to_Horizon(self) -> list:
        """Retrouve les indices correspondant entre S0 et le Forecast.

        Returns:
            list: index entre S0 et S0 + horizon
        """

        df_after_s0 =  self.h_data.loc[self.S0_index:]

        self.horizon = min(len(df_after_s0), self.horizon)

        indexes = df_after_s0.iloc[:self.horizon].index.to_list() 
        # faire un [1:self.horizon] pour résoudre le pb de s0 dans le forecast
        
        return indexes


    @abstractmethod
    def forecasting(self):
        pass

    # Plot methods
    def plot_forcast(self, df : pd.DataFrame, forecast_cols : list):

        plt.figure(figsize=(13,7))

        plt.plot(df["Log Price"], c="black")

        for c in forecast_cols:
            if c in df.columns:
                plt.plot(df[c], c=np.random.rand(3,))
        
        plt.title(self)
        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.grid(True, alpha = 0.7)
        plt.show()


class MonteCarlo(Forecast):
    """Hérite de Forecast.  
    Cette classe permet de simuler un grand nombre de fBm pour retourner le fbm moyen.  
    Mathématiquement cette méthode ne peut pas être utilisé car le forecast final sera S0 etant donné que l'espérance de nos v.a. est de 0.

    Args:
        Forecast (class)
    """

    def __init__(self,
                h_data : pd.DataFrame,
                h_freq : str,
                date : str,
                horizon : int,
                generations = 10000):
        
        super().__init__(h_data, h_freq, date, horizon)
        
        self.generation = generations

        self.paths = pd.DataFrame(index=self.indexes)


    def simulate_fbm_cholesky(self, epsilon = 1e-10) -> np.ndarray :
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

    def forecasting(self) -> np.ndarray:
        """Simule des fbm transformés avec l'odre de grandeur des prix correspondant.

        Returns:
            np.ndarray: Chemin moyen de simulations.
        """
        for i in range(self.generation):

            path = self.simulate_fbm_cholesky()

            self.paths[i] = self.s0 * np.exp(self.vol * (path - path[0]))

        return np.mean(self.paths, axis=0)

    
class CovarianceBased(Forecast):
    """
    Hérite de l'objet Forecast.  
    Permet le forecast grace à la matrice de covariance des Log Price et des propriétés de dépendance des fBm.   
    """

    def __init__(self,
                h_data : pd.DataFrame,
                h_freq : str,
                date : str,
                horizon : int
                ):
        
        cut_df = h_data.loc[:date] # Conserver la matrice avant la date à partir de laquelle on veut forecaster

        super().__init__(cut_df, h_freq, date, horizon)

        # Attributs nécéssaires pour le forecast

        self.t_n = len(cut_df)

        self.sigma_Y = self.get_sigma_Y()

        self.sigma_XY = self.get_sigma_XY()


    def get_cov_fbm(self, sigma : float, s : int, t: int, hurst : float) -> float: 
        """
        Définition du fBm. Covariance entre deux v.a. s et t avec s < t

        Args:
            sigma (float): Volatilité.
            s (int): s < t
            t (int): t > s
            hurst (float): Hurst exponent

        Returns:
            float: Covariance entre Xs et Xt : E[XsXt]
        """
        return (
            (pow(sigma, 2) / 2) * (
            np.abs(s)**(2*hurst) + np.abs(t)**(2*hurst) - np.abs(s-t)**(2*hurst))
            )


    def get_sigma_Y(self) -> np.ndarray:
        """Matrice de covariance de notre objet.  
        Utilise get_cov_fbm

        Returns:
            np.ndarray: Tableau à 2 dimensions.
        """

        res = np.ndarray((self.t_n, self.t_n))

        for i in range(1, res.shape[0]+1):
            for j in range(1, res.shape[1]+1):
                res[i-1,j-1] = self.get_cov_fbm(self.vol, i, j, self.H)

        return res        


    def get_sigma_XY(self) -> np.ndarray:
        """Calcule le tableau de covariance entre X_t et X_t+h.  
        Utilise get_cov_fbm.

        Returns:
            np.ndarray: Tableau 1 dimension.
        """

        t_plus_h = self.t_n + self.horizon

        return np.array([self.get_cov_fbm(self.vol, t_plus_h, i, self.H) for i in range(1, self.t_n + 1)])


    def get_sigma_X(self, t= 0) -> float:
        """Permet de calculer le MSE.

        Args:
            t (int, optional): Defaults to 0.

        Returns:
            float.
        """
        return (pow(self.vol, 2) / 2) * (2 * np.abs(t + self.horizon)**(2*self.H)) 


    def forecasting(self) -> float:
        """Calcule matriciel entre sigmaXT.T, sigmaY^^(-1), Log Price

        Returns:
            float: Forecast d'horizon h
        """
        
        xy = self.sigma_XY.reshape(-1,1)
        Y = self.h_data["Log Price"].to_numpy().reshape(-1,1)

        X_t_h_Y = xy.T @ np.linalg.inv(self.sigma_Y) @ Y

        return X_t_h_Y.item()
    


def single_random_forecast(df : pd.DataFrame,
                    h : str,
                    horizon : int,
                    max_lenght : int) -> pd.DataFrame:
    """Forecast entre 2 dates aléatoires.

    Args:
        df (pd.DataFrame): A une timeframe, dataset d'un asset.
        h (str): fréquence de hurst utilisée.
        horizon (int): horizon de forecast.
        max_lenght (int): maximale taille de la matrice de covariance.

    Returns:
        pd.DataFrame: Dataset avec le forecast.
    """

    if len(df) == 0:
        return None

    index_deb = np.random.randint(0, len(df) - max_lenght)
    index_end = index_deb + max_lenght

    df_max : pd.DataFrame = df.iloc[index_deb:index_end]

    dates = df_max.index

    df_max[f"Forecast Horizon {horizon}"] = None

    for i, date in enumerate(dates):

        if i > 1 and i + horizon < len(df_max):

            forecastobj = CovarianceBased(h_data=df_max,
                                        h_freq=h,
                                        date=date,
                                        horizon=horizon)
            
            predict_date = dates[i + horizon]

            try:
                df_max.loc[f"Forecast Horizon {horizon}",predict_date] = forecastobj.forecasting()
            except:
                continue

    return df_max[["Log Price", f"Forecast Horizon {horizon}"]]

def single_rolling_forecast(df : pd.DataFrame,
                     h_freq : str,
                     horizon : int,
                     max_size_mat : int) -> pd.DataFrame:
    """Ajoute la colonne Forecast calculée dynamiquement à la Dataframe. 


    Args:
        df (pd.DataFrame): A une timeframe, dataset avec les données d'un asset.
        h_freq (str): Fréquence sur laquelle se base le forecast.
        horizon (int): Horizon de forecast.
        max_size_mat (int): Taille maximale de la matrice de covariance. Utilisée pour le rolling(window).
    """
    
    res_dict = {key : None for key in df.index}

    i = 1

    while(i > 0 and i + horizon < len(df)):

        if i > max_size_mat:
            df_max = df.iloc[i - max_size_mat:]
        else:
            df_max = df

        current_date = df.index[i]

        forecast_date = df.index[i + horizon]        

        forecast = CovarianceBased(df_max,h_freq,current_date,horizon).forecasting()

        res_dict[forecast_date] = forecast
        
        i += 1
    
    df_res = pd.DataFrame.from_dict(res_dict,
                                    orient="index",
                                    columns=["Forecast"])

    res = pd.concat([df[["Log Price"]], df_res], axis=1)
  
    return res



def single_forecast(df : pd.DataFrame,
                    horizons : list,
                    max_lenght : int,
                    rollong_method = True) -> dict[str,dict[int, pd.DataFrame]]:
    """
    Applique les méthodes random ou rolling forecast en fonction de random_method.

    Args:
        df (pd.DataFrame) : Dataset de l'actif.
        horizon (list): Liste des horizons à forecast.
        max_lenght (int): maximale lenght pour le forecast.
        random_method (bool, optional): True utilise random_forecast, sinon Rolling. Defaults equal False.

    Returns:
        dict[str,dict[int, pd.DataFrame]]: dict[Hurst, [Horizon, Data_Forecast]]
    """

    frequencies = df.columns.to_list()
    h_fq = [fq.split("Hurst ")[1] for fq in frequencies if fq.startswith("Hurst")]


    hurst_horizon : dict[str, dict] = {h_freq : None for h_freq in h_fq}

    for h in h_fq:
        print(f"Hurst {h}")

        horizon_df : dict[int, pd.DataFrame] = {horiz : None for horiz in horizons}

        for horiz in horizons:

            print(f"Horizon {horiz}")

            df_forecast : pd.DataFrame = None

            if rollong_method:            
                df_forecast = single_rolling_forecast(df, h, horiz, max_lenght)
                            
            else:
                df_forecast = single_random_forecast(df, h, horiz, max_lenght)
            
            horizon_df[horiz] = df_forecast
        
        hurst_horizon[h] = horizon_df
    
    return hurst_horizon

def process_forecast(size_mat_horizons : dict[int, list], dir_data : str, dir_output : str):
    """Parcourt le fichier avec les fichiers Excels contenant les feuilles au nom de chaque timeframe, avec comme colonnes les Log Price et les distributions de Hurst.  
    Enregistre tous les forecast au chemin spécifié.


    Args:
        size_mat_horizons (dict[int, list]): Associations de taille de matrice de covariance maximale et d'horizons de forecast.
        dir_data (str): Chemin des du dossiers contenant les données.
        dir_output (str): Chemin du dossier d'enregistrement.
    """

    files = [f for f in os.listdir(dir_data) if f.endswith(".xlsx")]

    

    timeframes = ["Daily", "1min"]

    for tframe in timeframes:

        print(tframe)

        tframe_output = os.path.join(dir_output, tframe)

        # asset_dict : dict[str, dict] = {}

        for file in files:

            file_json = file.split(".xlsx")[0] + ".json"


            # Ne pas recalculer si le fichier existe déjà
            if(file_json in os.listdir(tframe_output)):
                print(f"{file_json} déjà calculé.")
                continue

            print(file)
            
            data_file_path = os.path.join(dir_data, file)

            df = pd.read_excel(data_file_path, sheet_name=tframe, index_col=0)

            # Réduire le forecast à 6000 points de données
            if(tframe=="1min"):
                try:
                    df=df.tail(6000)
                except NameError as e:
                    print(file, " : Non exécuté.")
                    print(e)
                    continue


            # Dictionnaire à remplir et à enregistrer
            size_frcst : dict[int, dict] = {}

            for i, horizons in size_mat_horizons.items():           

                print(f"Covariance Matrix Size : {i}")
                
                frcst = single_forecast(df, horizons, i)

                size_frcst[i] = frcst

            
            # asset_dict[file] = size_frcst 

            output_file = os.path.join(tframe_output, file_json)   

            save_forecast(output_file, size_frcst)
                


def save_forecast(output_path : str, dico : dict) -> None:
    """Enregistre le dictionnaire conentant une dataframe au chemin spécifié.

    Args:
        output_path (str): chemin du fichier.
        dico (dict): dictionnaire à enregistrer.
    """

    def convert_to_dict(obj) -> dict:
        """Méthode récursive pour détecter une DataFrame et la changer en dictionnaire.

        Returns:
            _type_: _description_
        """

        if isinstance(obj, pd.DataFrame):
            obj.index = obj.index.astype(str).to_list() # On change les index en string pour l'enregistrement.
            return obj.dropna().to_dict(orient="split")
        
        if isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}  # Récursivité
        
        return obj  # Retourner inchangé si ce n'est pas une Series

    with open(output_path, mode="w") as file:
        json.dump(convert_to_dict(dico), file)

def load_forecast(path_file:str) -> dict[int,dict[str,dict[int,pd.DataFrame]]]:
    """Charge le .json au chemin spécifié et reconstruit le dictionnaire récursivement avec la Dataframe. 

    Args:
        path_file (str): Chemin du fichier.

    Returns:
        dict: Dictionnaire conenant la DataFrame
    """

    with open(path_file, mode="r") as f:
        loaded_data = json.load(f)

    def reconstruct_dict(obj):

        if isinstance(obj, dict):
            # Reconstruction d'un DataFrame
            if "columns" in obj and "index" in obj and "data" in obj:                
                df = pd.DataFrame(data=obj["data"], index=pd.to_datetime(obj["index"]), columns=obj["columns"])
                df.rename(columns={[c for c in df.columns.to_list() if c.startswith("Forecast")][0] : "Forecast"}, inplace=True)
                return df
                
            
            # Reconstruction d'une Series
            elif "index" in obj and "values" in obj:                
                return pd.Series(data=obj["values"], index=pd.to_datetime(obj["index"]))
            
            else:
                return {k: reconstruct_dict(v) for k, v in obj.items()} # Récursivité pour les dictionnaires imbriqués

            
        return obj  
    
    return reconstruct_dict(loaded_data)




if __name__ == "__main__":

    def explore_dict(d:dict, depth=0):
        for key, value in d.items():
            print("  " * depth + f"- {key}")  # Indentation pour la lisibilité
            if isinstance(value, dict):  # Vérifie si c'est un sous-dictionnaire
                explore_dict(value, depth + 1)
            elif isinstance(value, pd.DataFrame):
                print(value.columns)

    def increment_keys(d):
        if isinstance(d, dict):
            new_dict = {}
            for key, value in d.items():
                try:
                    key = int(key)
                except:
                    pass
                new_key = key + 1 if isinstance(key, int) and isinstance(value, pd.DataFrame) else key
                new_dict[new_key] = increment_keys(value)
            return new_dict
        else:
            return d

    
    def test_forecast():


        sizeMat_horizons = {
            10 : [1, 2],
            30 : [1, 2, 3],
            50 : [1, 3, 5]
        }

        dir_data = r"Data\\Final Data - Copie"
    
        dir_output = r"Data\\Forecasting\\Covariance Based\\Single Forecast"
        
        process_forecast(size_mat_horizons=sizeMat_horizons,
                         dir_data=dir_data,
                         dir_output=dir_output)

    def test_load():

        dir_forecast_path = r"Data\\Forecasting\\Covariance Based\\Single Forecast\\Daily - Copie"

        files = [i for i in os.listdir(dir_forecast_path) if i.endswith(".json")]

        for file in files:
            print(file)

            file_path = os.path.join(dir_forecast_path, file)

            dic = load_forecast(file_path)

            # new_dict = increment_keys(dic)

            # save_forecast(dico=new_dict, output_path=file_path)

            explore_dict(dic)

    # test_forecast()
    test_load()

    pass
