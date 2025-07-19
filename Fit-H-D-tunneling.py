import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import quadrature
from sympy import N
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

# Extracting x, y values and errors for HExp
x_H, y_H, y_err_H = zip(*HExp)

# Convert to NumPy arrays
x_H = np.array(x_H)
y_H = np.array(y_H)
y_err_H = np.array(y_err_H)

# Multiply y values and errors by 2/3
y_H = 2/3 * y_H
y_err_H = 2/3 * y_err_H

# Extracting x, y values and errors for DExp
x_D, y_D, y_err_D = zip(*DExp)

# Convert to NumPy arrays
x_D = np.array(x_D)
y_D = np.array(y_D)
y_err_D = np.array(y_err_D)

# Multiply y values and errors by 2/3
y_D = 2/3 * y_D
y_err_D = 2/3 * y_err_D

# Constantes
hbar = 0.658
hbar_b = 6.58e-4
kb = 8.61e-2
# Parámetros (ejemplo, reemplazar con los reales)
omega0H, omega0B, VBD, gamma = 28, 28, 72, 140
T_vals = np.linspace(4, 13, 100)
Vd = VBD
omegaB = omega0B
T= T_vals

def lambda_n(omegaB, gamma):
    return (np.sqrt(gamma**2 + 4 * (omegaB / hbar)**2) - gamma) / 2

def u1(omegaB, gamma):
    return np.sqrt(gamma / (gamma + 2 * lambda_n(omegaB, gamma)))

def u12(omegaB, gamma):
    return gamma / (gamma + 2 * lambda_n(omegaB, gamma))

def Wollynes(omega0, omegaB, T, gamma):
    product = 1
    for n in range(1, 401):
        num = (omega0 / hbar)**2 + ((2 * np.pi) / hbar * (kb * 1e3) / T * n)**2 + gamma * ((2 * np.pi) / hbar * (kb * 1e3) / T * n)
        denom = -(omegaB / hbar)**2 + ((2 * np.pi) / hbar * (kb * 1e3) / T * n)**2 + gamma * ((2 * np.pi) / hbar * (kb * 1e3) / T * n)
        product *= num / denom
    return product

def Etheta(Vd, omegaB, gamma, theta):
    return Vd - (hbar * lambda_n(omegaB, gamma)) / np.pi * theta

def Itheta(Vd, omegaB, gamma, T):
    def integrand(theta):
        # Apply a check to prevent overflow in np.exp and np.cosh
        exp_term = -Etheta(Vd, omegaB, gamma, theta) / ((kb * 1e3) / T)
        
        # Prevent overflow in np.exp
        if np.abs(exp_term) > 700:  # Avoiding np.exp overflow
            return 0

        # Prevent overflow in np.cosh
        if np.abs(theta) > 500:  # np.cosh(theta) grows exponentially for large theta
            return 0

        return np.exp(exp_term) / (np.cosh(theta)**2)
    
    # Perform the integration over a limited range
    integral, _ = quad(integrand, -100, 100, epsabs=1e-3, limit=50)  # Reduced range of integration
    
    # Return the final result
    return 0.5 * np.exp(Vd / ((kb * 1e3) / T)) * integral

def Itheta23(Vd, omegaB, gamma, T, lower_limit):
    # Define the integrand function as a pure scalar function
    def integrand(Ei, Vd, omegaB, gamma, T):
        exp_term = np.exp(-Ei / ((kb * 1e3) / T))
        denom_term = 1 + np.exp((2 * np.pi) / (hbar * lambda_n(omegaB, gamma)) * (Vd - Ei))
        return exp_term / denom_term

    # Integrate using scipy.integrate.quad with all arguments passed as args
    integral, _ = quad(integrand, lower_limits[0], np.inf, args=(Vd, omegaB, gamma, T), epsabs=1e-3, limit=50, epsrel=1e-3, maxp1=100)
    
    # The final result
    return (1 / ((kb * 1e3) / T)) * np.exp(Vd / ((kb * 1e3) / T)) * integral

def EpsilonN(gamma, Vd, omega0, omegaB, T):
    product = 1
    for n in range(1, 401):
        term1 = ((omega0 / hbar)**2 + ((2 * np.pi) / hbar * (kb * 1e3) / T * n)**2 + gamma * ((2 * np.pi) / hbar * (kb * 1e3) / T * n))
        term2 = (((2 * np.pi) / hbar * (kb * 1e3) / T * n) + lambda_n(omegaB, gamma))
        denom = (((2 * np.pi) / hbar * (kb * 1e3) / T * n)**2 * (((2 * np.pi) / hbar * (kb * 1e3) / T * n) + lambda_n(omegaB, gamma) + gamma))
        product *= (term1 * term2) / denom
    return Itheta(Vd, omegaB, gamma, T) * product

def EpsilonNEq23(gamma, Vd, omega0, omegaB, T):
    product = 1
    for n in range(1, 401):
        term1 = ((omega0 / hbar)**2 + ((2 * np.pi) / hbar * (kb * 1e3) / T * n)**2 + gamma * ((2 * np.pi) / hbar * (kb * 1e3) / T * n))
        term2 = (((2 * np.pi) / hbar * (kb * 1e3) / T * n) + lambda_n(omegaB, gamma))
        denom = (((2 * np.pi) / hbar * (kb * 1e3) / T * n)**2 * (((2 * np.pi) / hbar * (kb * 1e3) / T * n) + lambda_n(omegaB, gamma) + gamma))
        product *= (term1 * term2) / denom
    return Itheta23(Vd, omegaB, gamma, T, lower_limits) * product

def GammaClassic2j(omega0, omegaB, Vb, T, gamma):
    return (2 * omega0) / (2 * np.pi * hbar) * 1e12 * (lambda_n(omegaB, gamma) / (omegaB / hbar)) * np.exp(-Vb / ((kb * 1e3) / T))

def NGammaQ2j(omega0, omegaB, Vb, T, gamma):
    return GammaClassic2j(omega0, omegaB, Vb, T, gamma) * EpsilonN(gamma, Vb, omega0, omegaB, T)

def NGammaQ2jEq23(omega0, omegaB, Vb, T, gamma):
    return GammaClassic2j(omega0, omegaB, Vb, T, gamma) * EpsilonNEq23(gamma, Vb, omega0, omegaB, T)

def GammaQ2j(omega0, omegaB, Vb, T, gamma):
    return GammaClassic2j(omega0, omegaB, Vb, T, gamma) * Wollynes(omega0, omegaB, T, gamma)
    
T = 80  # Example temperature value (modify as needed)
omega0H, omega0B, VBD, gamma = 28, 28, 72, 140
# Example usage: Sweep lower limits and observe changes
lower_limits = [1e-8, 1e-7, 1e-6, 1e-5]
# results = [Itheta23(VBD, omegaB, 140, T, lower_limit) for lower_limit in lower_limits]

# print("Results for different lower limits:")
# for limit, result in zip(lower_limits, results):
#     print(f"Lower Limit: {limit}, Result: {result}")

# # Define the integrand function outside of Itheta23
# def integrand(Ei, Vd, omegaB, gamma, T):
#     exp_term = np.exp(-Ei / ((kb * 1e3) / T))
#     denom_term = 1 + np.exp((2 * np.pi) / (hbar * lambda_n(omegaB, gamma)) * (Vd - Ei))
#     return exp_term / denom_term

# def integrand(theta):
#         return np.exp(-Etheta(Vd, omegaB, gamma, theta) / ((kb * 1e3) / T)) / (np.cosh(theta))**2
# # Given parameters


# Ei_vals = np.linspace(0, 1e-3, 1000)
# integrand_vals = np.array([integrand(Ei, VBD, omega0B, gamma, T) for Ei in Ei_vals])

# # Find the range covering 80% of the integrand values
# y_min = np.percentile(integrand_vals, 10)  # 10th percentile for the lower bound
# y_max = np.percentile(integrand_vals, 90)  # 90th percentile for the upper bound
# print(y_min, y_max)

# # Plotting
# fig, ax = plt.subplots(figsize=(6, 6))

# # Plot the integrand values with a label for the legend
# ax.plot(Ei_vals, integrand_vals, label=f'omega0B={omega0B}, gamma={gamma}')

# # Set y-axis limits to cover the middle 80% of the values
# ax.set_ylim([y_min, y_max])

# # Add the legend
# ax.legend()

# # Set labels
# ax.set_xlabel('Ei')
# ax.set_ylabel('Integrand value')


###################  ALTAS TEMPERATURAS Plot con las curvas teóricas de la formula de Eli ################### 
# plt.show()
# Compute theory1 and theory2 for each temperature individually
# Compute theory1 and theory2 for each temperature individually
theory1 = np.array([NGammaQ2j(omega0H, omega0B, VBD, T, gamma) for T in T_vals])
theory2 = np.array([NGammaQ2j(omega0H / np.sqrt(2), omega0B / np.sqrt(2), VBD, T, gamma / 2) for T in T_vals])

# Create the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_yscale("log")  # Set the y-axis to log scale

# Plot experimental data with error bars
ax.errorbar(x_H, y_H, yerr=y_err_H, fmt='o', color='black', markersize=8, label="Exp-H")  # Black circles for H
ax.errorbar(x_D, y_D, yerr=y_err_D, fmt='^r', markersize=8, label="Exp-D")  # Red triangles for D

# Plot the theoretical curves
ax.plot(T_vals, theory1, label=r'$\Gamma_{SD,usc}-H$', color='blue', linewidth=3)  # Green line for H-Q
ax.plot(T_vals, theory2, label=r'$\Gamma_{SD,usc}-D$', color='purple', linewidth=3)  # Orange line for D-Q

# Labels and ticks
ax.set_xlabel(r"$1000/T$ ($K^{-1}$)", fontsize=18, fontname='serif')
ax.set_ylabel(r"Hopping rate [$s^{-1}$]", fontsize=18, fontname='serif')
ax.tick_params(axis='both', labelsize=18)

# Adjust x-axis ticks to match the xlim range
ax.xaxis.set_major_locator(ticker.FixedLocator([4, 5, 6, 7]))  # Adjusted to match xlim range
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor labels if cluttered

# Grid and limits
ax.grid(False)
ax.set_xlim(4, 7.5)
ax.set_ylim(1*10**10, 1.5*10**11)
# Add top x-axis
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())  # Correct way to set the top axis limits
ax_top.set_xticks([1000/250, 1000/180, 1000/100, 1000/80])
ax_top.set_xticklabels(["250", "180", "100", "80"], fontsize=18, fontname='serif')
ax_top.set_xlabel("T (K)", fontsize=18, fontname='serif')

# Adjust layout to avoid clipping ylabel
fig.tight_layout()

# Save figure as PDF
plt.savefig("High-Tem-Friction.pdf", dpi=300, bbox_inches='tight')
#plt.show()
plt.close()


###################  Plot con las curvas teóricas de la formula de Eli ################### 
# plt.show()
# Compute theory1 and theory2 for each temperature individually
# Compute theory1 and theory2 for each temperature individually
theory1 = np.array([NGammaQ2j(omega0H, omega0B, VBD, T, gamma) for T in T_vals])
theory2 = np.array([NGammaQ2j(omega0H / np.sqrt(2), omega0B / np.sqrt(2), VBD, T, gamma / 2) for T in T_vals])

# Create the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_yscale("log")  # Set the y-axis to log scale

# Plot experimental data with error bars
ax.errorbar(x_H, y_H, yerr=y_err_H, fmt='bo', markersize=8, label="Exp-H")  # Blue circles for H
ax.errorbar(x_D, y_D, yerr=y_err_D, fmt='^r', markersize=8, label="Exp-D")  # Red triangles for D

# Plot the theoretical curves
ax.plot(T_vals, theory1, label=r'$\Gamma_{SD,usc}-H$', color='g', linewidth=3,)  # Green line for H-Q
ax.plot(T_vals, theory2, label=r'$\Gamma_{SD,usc}-D$', color='orange', linewidth=3)  # Purple line for D-Q
# Load the decay data from files
decay_rate_H = np.loadtxt('decay-H-Elena-fr140.dat')
decay_rate_D = np.loadtxt('decay-D-Elena-fr70.dat')

# Extract columns for the first dataset (decay_rate_H)
temperaturas_decay_H = decay_rate_H[:, 0]  # Temperatures
b_final_values_H = decay_rate_H[:, 1]  # Fitted values

# Extract columns for the second dataset (decay_rate_D)
temperaturas_decay_D = decay_rate_D[:, 0]  # Temperatures
b_final_values_D = decay_rate_D[:, 1]  # Fitted values

# Transform data for both datasets
x_decay_H = 1000 / temperaturas_decay_H  # Divide each temperature by 1000
b_final_values_H *= 1e12  # Scale b_final values for the first dataset

x_decay_D = 1000 / temperaturas_decay_D  # Divide each temperature by 1000 for the second dataset
b_final_values_D *= 1e12  # Scale b_final values for the second dataset

# Add both datasets to the plot
ax.plot(x_decay_H, b_final_values_H, 's', color='purple', markersize=7, label="NS-H")
ax.plot(x_decay_D, b_final_values_D, '*', color='black', markersize=12, label="NS-D")

# Labels and ticks
ax.set_xlabel(r"$1000/T$ ($K^{-1}$)", fontsize=18, fontname='serif')
ax.set_ylabel(r"Hopping rate [$s^{-1}$]", fontsize=18, fontname='serif')
ax.tick_params(axis='both', labelsize=18)
ax.xaxis.set_major_locator(ticker.FixedLocator([0, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor labels if cluttered
ax.grid(False)
ax.legend()

# Add top x-axis
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks([1000/250, 1000/180, 1000/100, 1000/80])
ax_top.set_xticklabels(["250", "180", "100", "80"], fontsize=18, fontname='serif')
ax_top.set_xlabel("T (K)", fontsize=18, fontname='serif')

# Adjust layout to avoid clipping ylabel
fig.tight_layout()

# Save figure as PDF
plt.savefig("Tunneling-H-D-Analitic.pdf", dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

###################  Plot con solo el EXP y las SIMULACIONES NUMERICAS ################### 

# Create the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_yscale("log")  # Set the y-axis to log scale

# Plot experimental data with error bars
ax.errorbar(x_H, y_H, yerr=y_err_H, fmt='o', color='red', markersize=6, label="Exp")  # Blue circles for H


# Load the decay data from files
decay_rate_H = np.loadtxt('decay-H-Elena-fr140.dat')
decay_rate_D = np.loadtxt('decay-D-Elena-fr70.dat')

# Extract columns for the first dataset (decay_rate_H)
temperaturas_decay_H = decay_rate_H[:, 0]  # Temperatures
b_final_values_H = decay_rate_H[:, 1]  # Fitted values

# Extract columns for the second dataset (decay_rate_D)
temperaturas_decay_D = decay_rate_D[:, 0]  # Temperatures
b_final_values_D = decay_rate_D[:, 1]  # Fitted values

# Transform data for both datasets
x_decay_H = 1000 / temperaturas_decay_H  # Divide each temperature by 1000
b_final_values_H *= 1e12  # Scale b_final values for the first dataset

x_decay_D = 1000 / temperaturas_decay_D  # Divide each temperature by 1000 for the second dataset
b_final_values_D *= 1e12  # Scale b_final values for the second dataset

# Add both datasets to the plot
ax.plot(x_decay_H, b_final_values_H, 's', color='blue', markersize=7, label="NS")
# Labels and ticks
ax.set_xlabel(r"$1000/T$ ($K^{-1}$)", fontsize=18, fontname='serif')
ax.set_ylabel(r"Hopping rate [$s^{-1}$]", fontsize=18, fontname='serif')
ax.tick_params(axis='both', labelsize=18)
ax.xaxis.set_major_locator(ticker.FixedLocator([0, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor labels if cluttered
ax.grid(False)
ax.legend(fontsize=18)
ax.text(8, 10**9, "(a)", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")
ax.text(8, 0.7*10**11, "H", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")
# Add top x-axis
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks([1000/250, 1000/180, 1000/100, 1000/80])
ax_top.set_xticklabels(["250", "180", "100", "80"], fontsize=18, fontname='serif')
ax_top.set_xlabel("T (K)", fontsize=18, fontname='serif')

# Adjust layout to avoid clipping ylabel
fig.tight_layout()

# Save figure as PDF
plt.savefig("figure-tunneling-H.pdf", dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

# Create the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_yscale("log")  # Set the y-axis to log scale

ax.errorbar(x_D, y_D, yerr=y_err_D, fmt='^r', markersize=6, label="Exp")  # Red triangles for D
ax.plot(x_decay_D, b_final_values_D, '*', color='blue', markersize=12, label="NS")
# Labels and ticks
ax.set_xlabel(r"$1000/T$ ($K^{-1}$)", fontsize=18, fontname='serif')
ax.set_ylabel(r"Hopping rate [$s^{-1}$]", fontsize=18, fontname='serif')
ax.tick_params(axis='both', labelsize=18)
ax.xaxis.set_major_locator(ticker.FixedLocator([0, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor labels if cluttered
ax.grid(False)
ax.legend(fontsize=18)
ax.text(8, 10**9, "(b)", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")
ax.text(8, 0.7*10**11, "D", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")
# Add top x-axis
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks([1000/250, 1000/180, 1000/100, 1000/80])
ax_top.set_xticklabels(["250", "180", "100", "80"], fontsize=18, fontname='serif')
ax_top.set_xlabel("T (K)", fontsize=18, fontname='serif')

# Adjust layout to avoid clipping ylabel
fig.tight_layout()

# Save figure as PDF
plt.savefig("figure-tunneling-D.pdf", dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

# Cargar los datos
decay_rate_H = np.loadtxt('decay-H-Elena-fr140.dat')
decay_rate_D = np.loadtxt('decay-D-Elena-fr70.dat')

# Transformar datos
x_decay_H = 1000 / decay_rate_H[:, 0]
b_final_values_H = decay_rate_H[:, 1] * 1e12

x_decay_D = 1000 / decay_rate_D[:, 0]
b_final_values_D = decay_rate_D[:, 1] * 1e12

# Crear figura
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_yscale("log")

# Graficar datos experimentales
ax.errorbar(x_H, y_H, yerr=y_err_H, fmt='o', color='red', markersize=6, label="Exp-H")
ax.errorbar(x_D, y_D, yerr=y_err_D, fmt='^', color='red', markersize=6, label="Exp-D")

# Graficar datos ajustados
ax.plot(x_decay_H, b_final_values_H, 's', color='blue', markersize=7, label="NS-H")
ax.plot(x_decay_D, b_final_values_D, '*', color='blue', markersize=12, label="NS-D")

# Etiquetas y estilo
ax.set_xlabel(r"$1000/T$ ($K^{-1}$)", fontsize=18, fontname='serif')
ax.set_ylabel(r"Hopping rate [$s^{-1}$]", fontsize=18, fontname='serif')
ax.tick_params(axis='both', labelsize=18)
ax.xaxis.set_major_locator(ticker.FixedLocator([0, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(False)
ax.legend(fontsize=14)

# Etiquetas en el gráfico
#ax.text(8, 0.7*10**11, "H", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")
#ax.text(8, 0.2*10**11, "D", fontsize=22, fontfamily="serif", fontweight="bold", ha="center")

# Eje superior (temperatura en K)
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks([1000/250, 1000/180, 1000/100, 1000/80])
ax_top.set_xticklabels(["250", "180", "100", "80"], fontsize=18, fontname='serif')
ax_top.set_xlabel("T (K)", fontsize=18, fontname='serif')

# Ajuste y guardado
fig.tight_layout()
plt.savefig("figure-tunneling-H-D-joint.pdf", dpi=300, bbox_inches='tight')
plt.close()


#Funcion para obtener los radios entre los NS y el EXP con T mas cercana
def compute_ratio_nearest(x1, y1, x2, y2):
    """
    For each point in (x1, y1), find the closest x in x2 and compute y1 / y2.
    Returns the ratios and the corresponding matched x2 values.
    """
    ratios = []
    matched_x2 = []
    for xi, yi in zip(x1, y1):
        idx = np.argmin(np.abs(x2 - xi))  # Index of closest x2 to xi
        ratio = yi / y2[idx]
        ratios.append(ratio)
        matched_x2.append(x2[idx])
    return np.array(ratios), np.array(matched_x2)

ratios_H, matched_xH = compute_ratio_nearest(x_decay_H, b_final_values_H, x_H, y_H)
for xd, match_x, ratio in zip(x_decay_H, matched_xH, ratios_H):
    print(f"x_decay_H: {1000/xd:.3f}, Closest x_H: {1000/match_x:.3f}, Ratio: {ratio:.3f}")
np.savetxt("ratios_decayH_vs_expH.dat", np.column_stack((x_decay_H, matched_xH, ratios_H)),
           header="x_decay_H   closest_x_H   ratio_bfinal_over_yH")

print("x_decay_H\tb_final_values_H")
for x, b in zip(x_decay_H, b_final_values_H):
    print(f"{1000/x:.6f}\t{b:.6e}")