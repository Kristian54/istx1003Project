import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

ikke_varemerke = np.array(['Architecture', 'Classic', 'Creator 3-in-1', 'Creator Expert',
       'Friends', 'Hidden Side', 'Juniors', 'City', 'NINJAGO','Technic']
)

varemerke = np.array(['Batman', 'DC', 'Disney', 'Harry Potter', 'Jurassic World',
       'LEGO Frozen 2', 'Marvel', 'Minecraft', 'Minions', 'Monkie Kid',
       'Overwatch', 'Powerpuff Girls', 'Spider-Man', 'Star Wars',
       'Stranger Things', 'THE LEGO MOVIE 2', 'Trolls World Tour',
       'Unikitty'])

diverse = np.array(['DOTS', 'DUPLO', 'Ideas', 'Powered UP', 'Minifigures',
       'Speed Champions', 'Xtra', 'BrickHeadz'])

def getUncleanedData():
    return pd.read_csv("Data/Data/lego.population.csv", sep = ",", encoding = "latin1")

def getCleanData():
    df = getUncleanedData()
    # fjerner forklaringsvariabler vi ikke trenger
    df2 = df[['Set_Name', 'Theme', 'Pieces', 'Price', 'Pages', 'Unique_Pieces']]

    # fjerner observasjoner med manglende datapunkter
    df2 = df2.dropna()

    # gjør themes om til string og fjern alle tegn vi ikke vil ha med
    df2['Theme'] = df2['Theme'].astype(str)
    df2['Theme'] = df2['Theme'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex=True)

    # fjerner dollartegn og trademark-tegn fra datasettet
    df2['Price'] = df2['Price'].str.replace('\$', '', regex=True)

    # og gjør så prisen om til float
    df2['Price'] = df2['Price'].astype(float)

    # det er dataset dere skal bruke!
    return df2

def groupByTheme(df):
    varemerke_data = df[df['Theme'].isin(varemerke)]
    ikke_varemerke_data = df[df['Theme'].isin(ikke_varemerke)]
    diverse_data = df[df['Theme'].isin(diverse)]

    return varemerke_data, ikke_varemerke_data, diverse_data

clean_data = getCleanData()
varemerke_data, ikke_varemerke_data, diverse_data = groupByTheme(clean_data)

print(varemerke_data)
print(ikke_varemerke_data)
print(diverse_data)
