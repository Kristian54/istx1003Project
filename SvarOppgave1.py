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

#print("Varemerkedata:\n", varemerke_data)
#print("Ikke-varemerkedata:\n", ikke_varemerke_data)
#print("Diverse:\n", diverse_data)

############################################

# Viser kryssplott for varemerke, ikke-varemerke og diverse data. Har ikke regresjonslinje.

# plt.scatter(varemerke_data['Pieces'], varemerke_data['Price'])
# plt.title("Varemerke")
# plt.xlabel('Antall brikker')
# plt.ylabel('Pris i dollar [$]')
# plt.gca().set_aspect(5)
# plt.show()
#
# plt.scatter(ikke_varemerke_data['Pieces'], ikke_varemerke_data['Price'])
# plt.title("Ikke-varemerke")
# plt.xlabel('Antall brikker')
# plt.ylabel('Pris i dollar [$]')
# plt.gca().set_aspect(5)
# plt.show()
#
# plt.scatter(diverse_data['Pieces'], diverse_data['Price'])
# plt.title("Diverse")
# plt.xlabel('Antall brikker')
# plt.ylabel('Pris i dollar [$]')
# plt.show()

############################################

# Viser kryssplott med regresjonslinje for varemerke, ikke-varemerke og diverse data.
# Printer også OLS-resultatet for hver av modellene.

# Varemerke
formel = 'Price ~ Pieces'

modell = smf.ols(formel, data = varemerke_data)
resultat = modell.fit()
varemerke_resultat = resultat
print("Varemerkedata:\n", resultat.summary())

slope = resultat.params['Pieces']
intercept = resultat.params['Intercept']

regression_x = np.array(varemerke_data['Pieces'])
regression_y = slope * regression_x + intercept
plt.scatter(varemerke_data['Pieces'], varemerke_data['Price'], label='Data Points')
plt.plot(regression_x, regression_y, color='red', label='Regression Line')
plt.xlabel('Antall brikker')
plt.ylabel('Pris [$]')
plt.title('Varemerkedata - Kryssplott med regresjonslinje (enkel LR)')
plt.legend()
plt.grid()
plt.show()

# Ikke-varemerke
modell = smf.ols(formel, data = ikke_varemerke_data)
resultat = modell.fit()
ikke_varemerke_resultat = resultat
print("Ikke-varemerkedata\n", resultat.summary())
regression_x = np.array(ikke_varemerke_data['Pieces'])
plt.scatter(ikke_varemerke_data['Pieces'], ikke_varemerke_data['Price'], label='Data Points')
plt.plot(regression_x, slope * regression_x + intercept, color='red', label='Regression Line')
plt.xlabel('Antall brikker')
plt.ylabel('Pris [$]')
plt.title('Ikke-varemerkedata - Kryssplott med regresjonslinje (enkel LR)')
plt.legend()
plt.grid()
plt.show()

# Diverse
modell = smf.ols(formel, data = diverse_data)
resultat = modell.fit()
diverse_resultat = resultat
print("Diverse-data:\n", resultat.summary())
resultat.summary()
regression_x = np.array(diverse_data['Pieces'])
plt.scatter(diverse_data['Pieces'], diverse_data['Price'], label='Data Points')
plt.plot(regression_x, slope * regression_x + intercept, color='red', label='Regression Line')
plt.xlabel('Antall brikker')
plt.ylabel('Pris [$]')
plt.title('Diverse data - Kryssplott med regresjonslinje (enkel LR)')
plt.legend()
plt.grid()
plt.show()

############################################

# Denne koden lager et kryssplott med regresjonslinjer for varemerke_data, ikke_varemerke_data og
# diverse_data.

resultater = []
for data, label in [(varemerke_data, 'Varemerke'), (ikke_varemerke_data, 'Ikke-varemerke'), (diverse_data, 'Diverse')]:
    modell = smf.ols('Price ~ Pieces', data = data)
    resultat = modell.fit()
    resultater.append(resultat)
    slope = resultat.params['Pieces']
    intercept = resultat.params['Intercept']
    regression_x = np.array(data['Pieces'])
    regression_y = slope * regression_x + intercept
    scatter = plt.scatter(data['Pieces'], data['Price'], label=label)
    plt.plot(regression_x, regression_y, label=f'Regression Line - {label}', color=scatter.get_edgecolor())

plt.xlabel('Antall brikker')
plt.ylabel('Pris')
plt.title('Kryssplott med regresjonslinjer')
plt.legend()
plt.grid()
plt.show()


# Concatenate the DataFrames and add a new column to indicate the category
varemerke_data['Category'] = 'Varemerke'
ikke_varemerke_data['Category'] = 'Ikke-varemerke'
diverse_data['Category'] = 'Diverse'

combined_data = pd.concat([diverse_data, varemerke_data, ikke_varemerke_data])

# Convert Category to a categorical variable
combined_data['Category'] = combined_data['Category'].astype('category')

# Update the formula to include the new category column
# Bruker C() for å indikere at Category er en kategorisk variabel og sammenligner mot Ikke-varemerke
formula = 'Price ~ Pieces + C(Category, Treatment(reference="Ikke-varemerke"))'

# Fit the model using the combined data
model = smf.ols(formula, data=combined_data)
result = model.fit()
print("Combined data:\n", result.summary())


############################################

# Plotter residualer og QQ-plott for varemerke_data, ikke_varemerke_data og
# diverse_data.

# Varemerke
figure, axis = plt.subplots(1, 2, figsize = (15, 5))
sns.scatterplot(x = varemerke_resultat.fittedvalues, y = varemerke_resultat.resid, ax = axis[0])
axis[0].set_ylabel("Residual")
axis[0].set_xlabel("Predikert verdi")

sm.qqplot(varemerke_resultat.resid, line = '45', fit = True, ax = axis[1])
axis[1].set_ylabel("Kvantiler i residualene")
axis[1].set_xlabel("Kvantiler i normalfordelingen")
plt.title("Varemerke")
plt.show()

# Ikke-varemerke
figure, axis = plt.subplots(1, 2, figsize = (15, 5))
sns.scatterplot(x = ikke_varemerke_resultat.fittedvalues, y = ikke_varemerke_resultat.resid, ax = axis[0])
axis[0].set_ylabel("Residual")
axis[0].set_xlabel("Predikert verdi")

sm.qqplot(ikke_varemerke_resultat.resid, line = '45', fit = True, ax = axis[1])
axis[1].set_ylabel("Kvantiler i residualene")
axis[1].set_xlabel("Kvantiler i normalfordelingen")
plt.title("Ikke-varemerke")
plt.show()

# Diverse
figure, axis = plt.subplots(1, 2, figsize = (15, 5))
sns.scatterplot(x = diverse_resultat.fittedvalues, y = diverse_resultat.resid, ax = axis[0])
axis[0].set_ylabel("Residual")
axis[0].set_xlabel("Predikert verdi")

sm.qqplot(diverse_resultat.resid, line = '45', fit = True, ax = axis[1])
axis[1].set_ylabel("Kvantiler i residualene")
axis[1].set_xlabel("Kvantiler i normalfordelingen")
plt.title("Diverse")
plt.show()