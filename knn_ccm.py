# import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# reading input from excel file
df1 = pd.read_excel("Input.xlsx", sheet_name = 'Financial_Data', index_col = 0)

df1.dropna(inplace = True)

df1['Enterprise_Value/Revenue'] = df1['Enterprise Value']/df1['CY Total Revenue']
df1['Enterprise_Value/EBITDA'] = df1['Enterprise Value']/(df1['CY Total Revenue'] * df1['CY EBITDA Margin %'])

# selecting training data
df2 = df1[:-1]
variables = df2.columns.tolist()

x_variables = variables[:-3]
y_variables = variables[-2:]

x_train = np.array(df2[x_variables])
y_train = np.array(df2[y_variables])

# scaling inputs
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

# number of neighbours
n = np.linspace(1,len(df2),len(df2)).astype(int)

# training models
models = {}

for i in n:
    models['model_'+str(i)] = KNeighborsRegressor(n_neighbors=i, weights='distance')
    models['model_'+str(i)].fit(x_train_scaled, y_train)

# selecting data of subject company
df3 = df1[-1:]

x_pred = np.array(df3[x_variables])

x_pred_scaled = scaler.transform(x_pred.reshape((1,len(x_variables))))

# predicting multiples of subject company
results = pd.DataFrame(index = y_variables)

for i in n: 
    results["N = "+str(i)] = models['model_'+str(i)].predict(x_pred_scaled).flatten()

# reading excel file
df_factors = pd.read_excel("Input.xlsx", sheet_name='Factors', index_col=0)

# droping na values
df_factors.dropna(axis = 1, inplace = True)

# selecting adjustment factor
df_factors['Adj'] = np.where(df_factors['Risk'] == 'No', 0, np.where(df_factors['Risk'] == 'Low', 0.05, 
                                                                     np.where(df_factors['Risk'] == 'Medium', 0.1, 0.15)))

total_adj = df_factors['Adj'].sum()

# adjusted multiples of subject company
adj_results = results * (1 - total_adj)

# calculating EBITDA for latest financial/calendar year
df1['CY EBITDA'] = df1['CY Total Revenue'] * df1['CY EBITDA Margin %']

# calculating revenue CAGR over the last 5-years
df1['Revenue CAGR'] = (((1 + df1['CY Total Revenues, 1 Yr Growth %']) * (1 + df1['CY-1 Total Revenues, 1 Yr Growth %']) * 
                        (1 + df1['CY-2 Total Revenues, 1 Yr Growth %']) *  (1 + df1['CY-3 Total Revenues, 1 Yr Growth %']) * 
                        (1 + df1['CY-4 Total Revenues, 1 Yr Growth %']))) ** 0.2 - 1

df_format = df1[['Enterprise Value','CY Total Assets','CY Total Revenue','CY EBITDA','CY EBITDA Margin %',
                 'CY Net Income Margin %','CY Return on Assets %','Revenue CAGR']].iloc[:-1]

# droping columns containing no value
df_format.dropna(inplace = True)

#calculating number of rows
num = len(df_format)

# changing format
df_format = df_format.transpose()

# calculating minimum, maximum, average, median
df_format['High'] = df_format.iloc[:,:num].max(axis = 1)
df_format['Low'] = df_format.iloc[:,:num].min(axis = 1)
df_format['Average'] = df_format.iloc[:,:num].mean(axis = 1)
df_format['Median'] = df_format.iloc[:,:num].median(axis = 1)

# merge the subject company
merged_df_format = pd.merge(df_format,df1.iloc[-1],left_index=True,right_index=True,how='inner')

# saving data to excel file
excel_file_path = 'Output.xlsx'
with pd.ExcelWriter(excel_file_path) as writer:
    merged_df_format.to_excel(writer, sheet_name='Financial Data', index = True)
    results.to_excel(writer, sheet_name='Multiples', index=True)
    df_factors.to_excel(writer, sheet_name='Factors', index=True)
    adj_results.to_excel(writer, sheet_name='Adjusted Multiples', index=True)
    
print(f"Data saved to '{excel_file_path}' with multiple sheets.")
