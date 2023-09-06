# %% [markdown]
# En este ejemplo estudiaremos el comportamiento de los sismos ocurridos 
# en Perú durante el período de 1960-2021. La información ha sido recopilada por el 
# Instituto Geofísico del Perú (IGP)

# %% [markdown]
# El presente trabajo se dividirá en 3 etapas:
# - 1ra Etapa : Limpieza y orden de los datos
# - 2da Etapa: Descripción de los datos
# - 3ra Etapa: Visualización gráfica de los datos

# %% [markdown]
# El link de referencia es el siguiente:
# https://www.datosabiertos.gob.pe/dataset/catalogo-sismico-1960-2021-igp

# %% [markdown]
# ---

# %% [markdown]
# OBTENCIÓN Y COMPRENSIÓN DE LOS DATOS

# %%
# Importamos la librería a utilizar.

import pandas as pd

# %%
# Extraemos los datos.

data = pd.read_excel("Catalogo1960_2021.xlsx")

# %%
# Consultamos los primeros 5 registros de nuestra base de datos.
                                                                                                                                        
data.head()

# %%
# Consultamos las dimensiones de nuestra base de datos.

# En este caso muestra una cantidad de 22 712 filas por 8 columnas.

data.shape

# %%
# Verificamos el tipo de dato de las diferentes columnas.

data.dtypes

# %% [markdown]
# ---

# %% [markdown]
# 1RA ETAPA: LIMPIEZA Y ORDEN DE LOS DATOS

# %%
# Todos los registros(filas) de la tabla cuentan con un identificado único, por lo que
# la columna "ID" es redundante.

# Procedemos a eliminar la columna ID y nombrar a la columna de identificadores de esta manera.

del data["ID"]
data.index.name = "ID"

# %%
# Tabla modificada sin la columna ID original

data.head()

# %%
# La columna FECHA_UTC es de tipo int(entero), por lo que la convertimos a tipo date(fecha).

# Para ello utilizamos la función .to_datetime de nuestra librería
# especificando el formato de año/mes/día (%Y%m%d).

data.FECHA_UTC = pd.to_datetime(data.FECHA_UTC,format="%Y%m%d")

# %%
# Formato de la columna FECHA_UTC modificada

data.head()

# %%
# Para poder trabajar con el tipo de dato time (tiempo), importamos la siguiente librería

import datetime as dt

# %%
# Mediante una exploración de los datos, encontramos que hay registros, como en este caso
# el de posición 73, que tienen una hora registrada de 210.

# En este tipo de casos,el formato de tiempo que le asignaremos sería de 0:02:10 

data.iloc[73]

# %%
# La siguiente función nos permitirá convertir números enteros en formato hora

# La función dt.time recibe de parámetro hora,minutos,segundos

# Dado que hay registros en los cuáles existen números del formato 210, la función dt.time
# dará un error.

# Está función descompone un número (x) de 2 en 2 cifras en un total de 6 cifras

# Así el número 210,mediante esta función, sería 0-2-10 (hora-minuto-segundo)

def intToTime(x):
    if (x<100):
        return dt.time(0,0,x)
    elif (100<= x<10000):
        return dt.time(0,x//100,x%100)
    else:
        return dt.time(x//10000,(x%10000)//100,(x%10000)%100)

# %%
# Aplicamos la función a la columna HORA_UTC mediante el método .apply 

data["HORA_UTC"] = data["HORA_UTC"].apply(intToTime)

# %%
# Confirmamos el cambio 

data.head()

# %%
# Corroboramos el registro con índice 73 y hora 210

data.iloc[73]

# %%
# Como último paso, eliminaremos la columna FECHA_CORTE, pues es irrelevante para el estudio

del data["FECHA_CORTE"]

# %%
# Vista final de nuestra base de datos

data

# %% [markdown]
# ---

# %% [markdown]
# 2DA ETAPA: DESCRIPCIÓN DE LOS DATOS

# %%
# Consultamos algunos datos numéricos relevantes, tales como los valores máximo y mínimo, 
# así como datos estadísticos, como la variación estándar y los percentiles

data.describe()

# %%
# Cantidad de Sismos por año

# Agrupamos los registros por año
sismosPorAño = data.groupby(by=data["FECHA_UTC"].dt.year).size()

# Mostramos el resultado
sismosPorAño


# %%
# Cantidad de Sismos por mes (desde 1960 - 2021)

# Agrupamos (groupby) los sismos de acuerdo al mes registrado
sismosPorMes = data.groupby(by=data["FECHA_UTC"].dt.month).size()

# Renombramos las filas
sismosPorMes.index = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Setiembre",
        "Octubre","Noviembre","Diciembre"]

# Mostramos el resultado       
sismosPorMes

# %%
# Para poder hallar la cantidad de sismos por mes y año haremos un paso previo

# Generamos una tabla con los años y meses de cada registro
sample = pd.DataFrame([data["FECHA_UTC"].dt.year , data["FECHA_UTC"].dt.month],
index=["Año","Mes"]).T

# Agregamos una columna "Num" cuya función será el de contador
sample["Num"] = 1

# Mostramos el resultado
sample

# %%
# Cantidad de sismos por mes y año

# Generamos una "tabla dinámica" con el método pivot_table
cantidadSismosMesAño = sample.pivot_table(index="Año",columns="Mes",values="Num",
aggfunc="count",fill_value=0)

# Asignamos nombres a las columnas
cantidadSismosMesAño.columns = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Setiembre",
        "Octubre","Noviembre","Diciembre"]

# Mostramos el resultado
cantidadSismosMesAño

# %%
# Año con la mayor cantidad de sismos
sismosPorAño[sismosPorAño == sismosPorAño.max()]

# %%
# Descripción de los valores mínimos, máximos y promedio

# La expresión data.iloc[:,2:] nos permite seleccionar todos los registros de todas
# las columnas a partir del índice 2 (latitud,longitud, profundidad, magnitud)
columnasDeTipoNumérico = data.iloc[:,2:]

# Hallamos los valores mínimos, máximos y el promedio de estas columnas en un nuevo marco de datos
pd.DataFrame([columnasDeTipoNumérico.min(),columnasDeTipoNumérico.max(),
            columnasDeTipoNumérico.mean()], index=["Mínimo","Máximo","Promedio"]).T

# %%
# Descripción de los valores mínimo, máximo y promedio de las magnitudes por año

# Agrupar la data de acuerdo al año registrado
dataGroupedByYear = data["FECHA_UTC"].dt.year

# Agrupamos las magnitudes de acuerdo al año ocurrido
magnitudesGroupedByYear = data["MAGNITUD"].groupby(by = dataGroupedByYear)

# Calculamos las magnitudes mínimas, máximas y el promedio por grupos (años)
magnitudesMinMaxProm = pd.DataFrame([  magnitudesGroupedByYear.min(),
                        magnitudesGroupedByYear.max(),
                        magnitudesGroupedByYear.mean()],
                        index=["Magnitud Mínima","Magnitud Máxima","Magnitud Promedio"]).T

# Mostramos el resultado
magnitudesMinMaxProm

# %%
# Podemos consultar aquellos sismos que ocurrieron a una profundidad de 0 m.s.n.m

data[data["PROFUNDIDAD"]==0]

# %%
# Consultamos aquellos sismos ocurridos "cercanos" a las coordenadas geográficas de Lima

# Coordenadas geográficas de Lima: 

# Longitud:  77.0282° O        Latitud:  12.0432° S 
# Longitud: -77.02824          Latitud: -12.04318

# Registros con una longitud y latitud distanciadas a las coordenadas de Lima a lo más por 0.1 
data[(abs(data["LONGITUD"]+77.02824)<0.1) & (abs(data["LATITUD"]+12.04318)<0.1)]

# %%
# Consideremos los sismos con magnitudes superior a  7 (terremotos)

terremotos = data[data["MAGNITUD"]>=7]
terremotos

# %%
# Hallamos el terremoto de mayor magnitud y sus datos

terremotos[terremotos["MAGNITUD"] == terremotos["MAGNITUD"].max()]

# %% [markdown]
# ---

# %% [markdown]
# 3ra Etapa: Visualización gráfica de los datos

# %%
# Importamos la librería a utilizar
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Gráfica del número de sismos por año

# Generamos la figura y los ejes con un tamaño conveniente 
fig,ax = plt.subplots(figsize=(15,4))

# Indicamos los datos a utilizar, así como un color de trazo y unos marcadores 
lineas = ax.stem(sismosPorAño)

# Asignamos un título 
ax.set_title("Número de sismos por año", loc="left",fontdict={'fontsize':20, 'fontweight':'bold','color':'#9400D3'})

# Asignamos etiquetas a los ejes
ax.set_xlabel("Años",fontsize=15, labelpad=10)
ax.set_ylabel("Cantidad",fontsize=15)

# Agregamos líneas horizontales para una mejor visualización
ax.grid(axis="y",color="gray",linestyle="dashed")

# Coloreamos el fondo de la figura y modificamos los marcadores de los ejes (ticks)
ax.set_facecolor("#E0FFFF")
ax.tick_params(width=2,length=5)

# Mostramos la figura
plt.show()

# %%
# Gráfica del número de sismos por mes de los años 1960 a 2021

# Generamos la figura base y los ejes
fig,ax = plt.subplots(figsize=(15,4))

# Indicamos los datos y el tipo de figura como barras (bar)
barras = ax.bar(sismosPorMes.index.str.upper(),sismosPorMes.array,color="#FF8F00")

# Agregamos formato a las etiquetas
ax.bar_label(barras, label_type="edge",color="#795548",fontweight="bold",fontsize=15)
ax.set_title("Número de sismos por mes",fontsize=20,loc="left",pad=15,color="#FF1744",fontweight="bold")

# Agregamos un color de fondo 
ax.set_facecolor("#FFE3BF")

# Agregamos formato a los ejes
ax.set_xlabel("Meses",fontsize=15,fontweight="bold",labelpad=10)
ax.set_ylim(top=2500)
ax.tick_params(axis="x",labelsize=11)

# Mostramos el resultado
plt.show()

# %%
# Gráfica de los valores promedio, mínimo y máximos de la magnitud de los sismos por año

# Generamos la figura y ejes
fig,ax = plt.subplots(figsize=(15,4))

# Indicamos los datos a utilizar
rectas = magnitudesMinMaxProm.plot(ax = ax, marker="o", markersize=3.5)

# Posicionamos la leyenda en el márgen derecho
ax.legend(loc = "right")

# Agregamos un título con un formato específico
ax.set_title("Valor promedio, mínimo y máximo de las magnitudes de los sismos por año",
loc="left",fontdict={'fontsize':14, 'fontweight':'bold','color':'#9400D3'})

# Nombramos el eje "X"
ax.set_xlabel("Años",fontsize=15,fontweight="bold",labelpad=10)

# Generamos una malla horizontal
ax.grid(axis="y", color="gray", linestyle="dashed")

# Mostramos la figura
plt.show()

# %%
# Hallamos las relaciones entre las variables numéricas

# Haremos uso de la función corr()

# Generamos la figura base
plt.figure()
sns.heatmap(columnasDeTipoNumérico.corr(), annot=True, cmap='jet')

# %%
# El gráfico anterior muestra que los únicos índices de correlación significativas son 
# en los pares Latitud-Longitud y Profundidad-Longitud

# %%
# Gráfica de Latitud-Longitud y Profundidad-Longitud

# LONGITUD - LATITUD

fig,ax = plt.subplots(1,2,sharex=True,figsize=(15,6))
ax[0].scatter(data["LONGITUD"],data["LATITUD"],s=2,c="#0EA0E2")
ax[0].set_xlabel("LONGITUD",fontdict={"fontsize":15,"fontweight":"bold","color":"#191970"},labelpad=10)
ax[0].set_ylabel("LATITUD",fontdict={"fontsize":15,"fontweight":"bold","color":"#191970"},labelpad=10)
ax[0].set_title("LONGITUD VS LATITUD",loc="left",fontdict={"fontsize":20,"fontweight":"bold","color":"#FFAA00"}, pad = 10)
ax[0].set_facecolor("#E1F3FB")
ax[0].grid(linestyle="dashed")


# LONGITUD - PROFUNDIDAD

ax[1].scatter(data["LONGITUD"],data["PROFUNDIDAD"]*-1,s=2,c="#7E57C2")
ax[1].set_xlabel("LONGITUD",fontdict={"fontsize":15,"fontweight":"bold","color":"#191970"},labelpad=10)
ax[1].set_ylabel("PROFUNDIDAD",fontdict={"fontsize":15,"fontweight":"bold","color":"#191970"},labelpad=10)
ax[1].set_title("LONGITUD VS PROFUNDIDAD",loc="left",fontdict={"fontsize":20,"fontweight":"bold","color":"#FFAA00"}, pad = 10)
ax[1].set_facecolor("#EFEAF7")
ax[1].grid(linestyle="dashed")
plt.show()

# %% [markdown]
# Procederemos a hallar las rectas que mejor se aproximan a los datos

# %% [markdown]
# LONGITUD - LATITUD

# %%
# Importamos la función de modelo lineal
from  sklearn import linear_model

# Generamos el modelo
modelLongLat = linear_model.LinearRegression()

# Agregamos los datos al modelo lineal
modelLongLat.fit(data["LONGITUD"].values.reshape(-1,1),data["LATITUD"].values)

# Calculamos las ordenadas de los puntos de la recta 
predictionLongLat = modelLongLat.predict(data["LONGITUD"].values.reshape(-1,1))

# La recta tendrá por ecuación ax + b, donde:
a = modelLongLat.coef_[0]
b = modelLongLat.intercept_ 

# Graficamos los resultados
fig,ax = plt.subplots()

# Gráfica de los puntos del tipo (Longitud,Latitud)
ax.scatter(data["LONGITUD"],data["LATITUD"],label="datos",s=2)

# Gráfica de la recta
ax.scatter(data["LONGITUD"],predictionLongLat,label="regresión Lineal",s=2)

# Leyenda
plt.legend(loc="upper right",markerscale=5,edgecolor="red")

# Agregamos el título,etiquetas a los ejes y un fondo
ax.set_title("LONGITUD VS LATITUD",loc="left")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_facecolor("#E1F3FB")

# Agregamos un texto con la ecuación de la recta 
plt.text(0.1,0.1,"Ecuación de la recta\n"+f"{a:.2f}"+"x"+f"{b:.2f}",
transform=ax.transAxes,bbox={"facecolor":"#FBE9E1","linestyle":"dashed"},fontsize=12)

# Mostramos los resultados
plt.show()



# %%
# Evaluaremos la calidad del modelo lineal

# Para ello importamos la función de error cuadrático medio
from sklearn.metrics import mean_squared_error

# %%
# Error cuadrático medio o mean squared error (mse)
mse = mean_squared_error(data["LATITUD"].values,predictionLongLat)

# Raíz del error cuadrático medio (rmse)
rmse = np.sqrt(mse)

# Coeficiente de determinación R2
r2 = modelLongLat.score(data["LONGITUD"].values.reshape(-1,1),data["LATITUD"].values)

pd.Series([mse,rmse,r2],
index=["Error cuadrático medio","Raíz error cuadrático medio","Coeficiente de determinación"])

# %% [markdown]
# El coeficiente de correlación indica que existe una correlación moderada (0.55) entre los 
# valores de Longitud y Latitud

# %% [markdown]
# LONGITUD - PROFUNDIDAD

# %%
# Generamos el modelo
modelLongProf = linear_model.LinearRegression()

# Agregamos los datos al modelo lineal
modelLongProf.fit(data["LONGITUD"].values.reshape(-1,1),(data["PROFUNDIDAD"]*-1).values)

# Calculamos las ordenadas de los puntos de la recta 
predictionLongProf = modelLongProf.predict(data["LONGITUD"].values.reshape(-1,1))

# La recta tendrá por ecuación mx+n, donde:
m = modelLongProf.coef_[0]
n = modelLongProf.intercept_ 

# Graficamos los resultados
fig,ax = plt.subplots()

# Gráfica de los puntos del tipo (Longitud,Latitud)
ax.scatter(data["LONGITUD"],data["PROFUNDIDAD"]*-1,label="datos",s=2)

# Gráfica de la recta
ax.scatter(data["LONGITUD"],predictionLongProf,label="regresión Lineal",s=2)

# Leyenda
plt.legend(loc="upper right",markerscale=5,edgecolor="red")

# Agregamos el título,etiquetas a los ejes y un fondo
ax.set_title("LONGITUD VS PROFUNDIDAD",loc="left")
ax.set_xlabel("Longitud")
ax.set_ylabel("Profundidad")
ax.set_facecolor("#E1F3FB")

# Agregamos un texto con la ecuación de la recta 
plt.text(.1,.1,"Ecuación de la recta\n"+f"{m:.2f}"+"x"+f"{n:.2f}",
transform=ax.transAxes,bbox={"facecolor":"#FBE9E1","linestyle":"dashed"},fontsize=12)

# Mostramos los resultados
plt.show()


# %%
# Evaluaremos la calidad del modelo lineal

# Error cuadrático medio o mean squared error (mse)
mse_2 = mean_squared_error((data["PROFUNDIDAD"]*-1).values,predictionLongProf)

# Raíz del error cuadrático medio (rmse)
rmse_2 = np.sqrt(mse_2)

# Coeficiente de determinación R2
r2_2 = modelLongProf.score(data["LONGITUD"].values.reshape(-1,1),(data["PROFUNDIDAD"]*-1).values)

pd.Series([mse_2,rmse_2,r2_2],
index=["Error cuadrático medio","Raíz error cuadrático medio","Coeficiente de determinación"])



# %% [markdown]
# El coeficiente de correlación (0.12) en este caso indica que la correlación entre las variables Longitud y Profundidad es inexistente.


