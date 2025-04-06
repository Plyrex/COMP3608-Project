import pandas as pd
import csv
import scipy


df_weather = pd.read_csv(r"weather data\F1 Weather(2023-2018).csv")
df_drivers = pd.read_csv(r"championship_data\drivers.csv")
df_drivers_standing = pd.read_csv(r"championship_data\driver_standings.csv")
df_constructors = pd.read_csv(r"championship_data\constructor_results.csv")

print(df_constructors)