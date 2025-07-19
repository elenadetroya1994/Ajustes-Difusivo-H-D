import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager  # Import the font manager

# Given data with errors (tu conjunto de datos original)
DExp = [
    (4.34395, 144922334859, 1.17214786395e10),
    (4.53503184713, 123256800847, 9.9691269715e9),
    (4.76433121019, 103142847740, 6.6712e9),
    (4.99363057325, 7.58450066424e10, 7.917628796599998e9),
    (5.26114649682, 6.55481606673e10, 5.880363178399998e9),
    (5.54777070064, 5.22506150381e10, 4.2260358823999977e9),
    (5.87261146497, 3.90411255877e10, 3.7909813308499985e9),
    (6.25477707006, 3.26611417814e10, 2.1124692489499989e9),
    (6.65605095541, 2.25068631366e10, 2.0190767597000008e9),
    (8.33757961783, 8.10771685379e9, 4.6248284723e8),
    (9.52229299363, 3.97377149802e9, 3.2131399182500005e8),
    (10.0191082803, 3.16643003295e9, 5.477843174549999e8),
    (10.5350318471, 2.28990117159e9, 3.9357665759000003e8),
    (11.1082802548, 1.76640001701e9, 2.8205832313e8),
    (11.7770700637, 1.45330597936e9, 4.86740773503e8)
]
HExp = [
    (4.28505624479, 146653422387, 0.095e11),
    (4.49105418586, 122298676448, 0.095e11),
    (4.67885992531, 136463826834, 0.085e11),
    (4.93659405316, 102037947219, 0.059e11),
    (5.19351663298, 94921988421.8, 0.054e11),
    (5.50251354459, 72287159422.9, 0.0419e11),
    (5.82835007777, 59209423906.1, 0.034e11),
    (6.61751290587, 33730649757, 0.022e11),
    (7.11458607293, 26649834251.7, 0.016e11),
    (8.29795835556, 12665526673.1, 0.11e10),
    (9.49844077578, 5720000000, 0.03e10),
    (9.97772751524, 5705222548.29, 0.03e10),
    (10.5090885865, 4300000000, 0.050e10),
    (11.0741965299, 3831465464.87, 0.03e10),
    (11.0919153285, 3212051530.96, 0.026e10),
    (11.7430473628, 2517300000, 0.035e10),
    (12.4851694915, 1540000000, 0.05e10)
]

# Extracting x, y values and errors
x, y, y_err = zip(*HExp)

# Convert to NumPy arrays
x = np.array(x)
y = np.array(y)
y_err = np.array(y_err)

# Multiply y values and errors by 2/3
factor = 2/3
y = [yi * factor for yi in y]
y_err = [yi_err * factor for yi_err in y_err]

# Compute secondary x-axis values (1000/x)
x_top = [1000/xi for xi in x]

# Create figure and axis
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_xlim(4, 14)
ax.set_ylim(0.1e9, 0.5e12)

# Plot data with error bars
ax.errorbar(x, y, yerr=y_err, fmt='o', markersize=6, linestyle='None', color='blue', label='Exp')

# Labels and ticks
ax.set_xlabel(r"$1000/T$ ($K^{-1}$)", fontsize=18, fontname='serif')
ax.set_ylabel(r"Tunneling rate [$s^{-1}$]", fontsize=18, fontname='serif')
ax.tick_params(axis='both', labelsize=18)
ax.xaxis.set_major_locator(ticker.FixedLocator([0, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor labels if cluttered
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add top x-axis
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks([1000/250, 1000/180, 1000/100, 1000/80])
ax_top.set_xticklabels(["250", "180", "100", "80"], fontsize=18, fontname='serif')
ax_top.set_xlabel("T (K)", fontsize=18, fontname='serif')

# Legend
legend = ax.legend(loc='upper right', fontsize=18, frameon=False, markerscale=1.5)

# Additional text annotation (Epilog equivalent)
ax.text(0.5, 0.9, '', transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center', fontname='serif')

# Cargar el archivo 'decay-Li-Elena.dat'
decay_data = np.loadtxt('decay-H-Elena-fr140.dat')

# Extraer las columnas de datos cargados (Temperaturas, b_final)
temperaturas_decay = decay_data[:, 0]  # Temperaturas
b_final_values = decay_data[:, 1]  # b_final (valores ajustados)

# Dividir cada temperatura entre 1000 para el nuevo valor de x
x_decay = 1000 / temperaturas_decay

# Multiplicar los valores de la segunda columna por 10^12
b_final_values = b_final_values * 1e12

# Para el gráfico, ahora añadimos los valores obtenidos del archivo decay-Li-Elena.dat
# Estos valores están relacionados con la temperatura (en x) y el valor b_final
# Add the new dataset with label "NS"
ax.plot(x_decay, b_final_values, 'ro', markersize=6, label="NS")

# Update the legend
ax.legend(loc='upper right', fontsize=18, frameon=False, markerscale=1.5)

# Save the figure as PDF
plt.savefig("Tunneling-H-fr140.pdf", dpi=300, bbox_inches='tight')

# Mostrar el gráfico con los valores añadidos
#plt.show()

# Valores de x, y y errores (tomados de tu ejemplo)
x = np.array([4.28505624479, 4.49105418586, 4.67885992531, 4.93659405316, 5.19351663298, 5.50251354459, 
              5.82835007777, 6.61751290587, 7.11458607293, 8.29795835556, 9.49844077578, 9.97772751524, 
              10.5090885865, 11.0741965299, 11.0919153285, 11.7430473628, 12.4851694915])

y = np.array([146653422387, 122298676448, 136463826834, 102037947219, 94921988421.8, 72287159422.9, 
              59209423906.1, 33730649757, 26649834251.7, 12665526673.1, 5720000000, 5705222548.29, 
              4300000000, 3831465464.87, 3212051530.96, 2517300000, 1540000000])

# Dividir los valores de y entre 10^12
y = y / 1e12

# Valores de x_top que quieres buscar
x_top_vals = [1000/220, 1000/180,1000/140, 1000/100,1000/90, 1000/80]

# Constantes k y a
k = 0.86
a = 2.77

# Calcular el valor de (1 - cos(k * a))
factor = 1 - np.cos(k*a*np.cos(np.pi/6))

# Función para encontrar los valores de y más cercanos
def find_closest_values(x_vals, x_target, y_vals):
    closest_indices = []
    closest_values = []
    for target in x_target:
        # Encuentra el índice del valor de x más cercano
        idx = np.argmin(np.abs(x_vals - target))
        closest_indices.append(idx)
        closest_values.append(y_vals[idx] * factor)  # Multiplicar por el factor
    return closest_indices, closest_values

# Llamar a la función
indices, modified_y = find_closest_values(x, x_top_vals, y)

# Mostrar los resultados
for i, val in zip(indices, modified_y):
    print(f"Valor más cercano de alpha: {val} correspondiente T: {1000/x[i]} ")
# Load the experimental data from files

# Load the experimental data from filesimport numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# Cargar datos experimentales
dataT90 = np.loadtxt("/home/elena/fit/H-Pt/Fig2-90K.dat")
dataT140 = np.loadtxt("/home/elena/fit/H-Pt/Fig2-140K-H.dat")
dataT220 = np.loadtxt("/home/elena/fit/H-Pt/Fig2-220K.dat")
decay_data = np.loadtxt("/home/elena/fit/H-Pt/decay-H-Elena-fr140.dat")

# Extraer valores experimentales
x_exp_90, y_exp_90 = dataT90[:, 0], dataT90[:, 1]
x_exp_140, y_exp_140 = dataT140[:, 0], dataT140[:, 1]
x_exp_220, y_exp_220 = dataT220[:, 0], dataT220[:, 1]

# Definir funciones teóricas
def theoretical_curve_90(x):
    return (
        -0.5126635188979902 * np.sin(x / 2) ** 2
        + 3.1044083765186405 * np.sin(x) ** 2
        + 1.49392490787727 * np.sin(3 * x / 2) ** 2
    )

def theoretical_curve_140(x):
    return (
        -47.223801516675714 * np.sin(x / 2) ** 2
        + 60.75353781820512 * np.sin(x) ** 2
    )

def theoretical_curve_220(x):
    return (
        2.9564615118542556e-11 * np.sin(x / 2) ** 2
        + 219.58096596486251 * np.sin(x) ** 2
    )


# Generar valores para las curvas teóricas
x_vals = np.linspace(0, 2, 1000)
y_vals_90 = theoretical_curve_90(x_vals)
y_vals_140 = theoretical_curve_140(x_vals)
y_vals_220 = theoretical_curve_220(x_vals)

# Crear figura con tres subgráficos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  

# --------- Subgráfico T = 90K ----------
ax1.plot(x_vals, y_vals_90, color="blue", linewidth=3, label="CE")
ax1.scatter(x_exp_90, y_exp_90, color="black", s=50, facecolors='none', label="Exp", linewidths=2)  
ax1.set_xlim(min(x_exp_90), max(x_exp_90))
ax1.set_ylim(0, max(y_exp_90))
ax1.set_xticks([0, 0.5, 1, 1.5, 2])
ax1.set_yticks([1, 2, 3, 4, 5, 6])
ax1.tick_params(labelsize=14, colors="black")
ax1.text(1, 0.5, "T=90K", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")
ax1.set_xlabel(r'$\Delta k \, [\mathrm{\AA}^{-1}]$', fontsize=18, fontfamily="serif", color="black")
ax1.set_ylabel(r'$\alpha [ns^{-1}]$', fontsize=18, fontfamily="serif", color="black")
ax1.grid(True, linestyle="--", alpha=0.5)

# --------- Subgráfico T = 140K ----------
ax2.plot(x_vals, y_vals_140, color="blue", linewidth=3, label="CE")
ax2.scatter(x_exp_140, y_exp_140, color="black", s=50, facecolors='none', label="Exp", linewidths=2)  
ax2.set_xlim(min(x_exp_140), max(x_exp_140))
ax2.set_ylim(0, 50)
ax2.set_xticks([0, 0.5, 1, 1.5, 2])
ax2.set_yticks([10, 20, 30, 40])
ax2.tick_params(labelsize=14, colors="black")
ax2.text(1, 5, "T=140K", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")
ax2.set_xlabel(r'$\Delta k \, [\mathrm{\AA}^{-1}]$', fontsize=18, fontfamily="serif", color="black")
ax2.grid(True, linestyle="--", alpha=0.5)

# --------- Subgráfico T = 220K ----------
ax3.plot(x_vals, y_vals_220, color="blue", linewidth=3, label="CE")
ax3.scatter(x_exp_220, y_exp_220, color="black", s=50, facecolors='none', label="Exp", linewidths=2)  
ax3.set_xlim(min(x_exp_220), max(x_exp_220))
ax3.set_ylim(0, 50)
ax3.set_xticks([0, 0.5, 1, 1.5, 2])
ax3.set_yticks([50, 100, 150, 200, 250])
ax3.tick_params(labelsize=14, colors="black")
ax3.text(1, 19, "T=220K", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")
ax3.set_xlabel(r'$\Delta k \, [\mathrm{\AA}^{-1}]$', fontsize=18, fontfamily="serif", color="black")
ax3.grid(True, linestyle="--", alpha=0.5)

factor1 = 1 - np.cos(k*a*np.cos(np.pi/6))

# --------- Extraer puntos de decay-H-Elena-fr140.dat ----------
for ax, T in zip([ax1, ax2, ax3], [90, 140, 220]):
    selected_point = decay_data[decay_data[:, 0] == T]
    
    if selected_point.size > 0 and selected_point.shape[1] >= 2:
        x_decay, y_decay = 0.86, (selected_point[0, 1]*factor1)*1e3  
        ax.scatter(x_decay, y_decay, color="red", s=50, marker="D", label="NS")
        print(f"Extracted value: {selected_point[0, 1], y_decay}")
    else:
        print(f"⚠️ Warning: No se encontró un punto válido para T={T}K en decay-H-Elena-fr140.dat")


# Actualizar las leyendas después de añadir los puntos extra
for ax in [ax1, ax2, ax3]:
    ax.legend(fontsize=18, loc="upper right", bbox_to_anchor=(0.4, 1.01), frameon=False, markerscale=1.5)

# Guardar la figura como archivo PDF
plt.savefig("/home/elena/fit/H-Pt/aplha-vs-dk-fr140.pdf", format="pdf", dpi=300, bbox_inches='tight')
