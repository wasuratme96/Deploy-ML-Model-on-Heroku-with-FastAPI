import pandas as pd
import numpy as np

import plotly.express as px


def plot_categorical_count(df:pd.DataFrame, column_count:str) -> None:
    count_df = pd.DataFrame(df.groupby([column_count]).size()).reset_index()
    count_df.rename(columns = {0 : 'Records Count'}, inplace = True)
    count_df.sort_values(by = 'Records Count', ascending= False, inplace = True)

    count_plot = px.histogram(count_df, x = column_count , y = "Records Count",
                              color = column_count, width= 700, height=400,
                              title= f"Count of {column_count}")
    count_plot.show()

def numerical_hist_plot(data:pd.DataFrame, numerical_col:str) -> None:
    histogram_plot = px.histogram(data, x = numerical_col,
                                  width = 700, height=400,
                                  title= f"Hitogram of {numerical_col}")
    histogram_plot.show()