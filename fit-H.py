import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
import time

#  Ficheros y tiempos de ajuste para T-220K
# datasets = [
#         
# Definir datasets manualmente
datasets = [ 
            {
                "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T80-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_50000.dat",
                "Name": "fsi-T80_N_10000.dat",
                "Temperature": "80",
                 "tc_p0": 20,
                "tmin_int": 20,
                "tmax_int": 50,
                "tm_ajuste":32,
                "t_max_exp": 200,
                "output": "/home/elena/fit/H-Pt/ajustes/fits-T80-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_50000.pdf"
            },{
                "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T90-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_20000.dat",
                "Name": "fsi-T90_N_500.dat",
                "Temperature": "90",
                 "tc_p0": 20,
                "tmin_int": 20,
                "tmax_int": 50,
                "tm_ajuste":40,
                "t_max_exp": 300,
                "output": "/home/elena/fit/H-Pt/ajustes/fits-T90-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_20000.pdf"
                }
            # , {
            #     "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T120-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_50000.dat",
            #     "Name": "fsi-T100_N_50000.dat",
            #     "Temperature": "100",
            #     "tc_p0": 10,
            #     "tmin_int": 10,
            #     "tmax_int": 50,
            #     "tm_ajuste":35,
            #     "t_max_exp": 200,
            #     "output": "/home/elena/fit/H-Pt/ajustes/fits-T120-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_50000.pdf"
            # }
             ,{
                "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T140-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_50000.dat",
                "Name": "fsi-T140_N_50000.dat",
                "Temperature": "140",
                "tc_p0": 20,
                "tmin_int": 20,
                "tmax_int": 60,
                "tm_ajuste":50,
                "t_max_exp": 90,
                "output": "/home/elena/fit/H-Pt/ajustes/fits-T140-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_50000.pdf"
            }, {
                "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T180-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_50000.dat",
                "Name": "fsi-T180_N_50000.dat",
                "Temperature": "180",
                "tc_p0": 25,
                "tmin_int": 25,
                "tmax_int": 60,
                "tm_ajuste":35,
                "t_max_exp": 100,
                "output": "/home/elena/fit/H-Pt/ajustes/fits1-T180-k0p86-N4096-x30-xa28-fr008-g3-dt70-w10_N_50000.pdf"
            },
            # {
            #     "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T220-k0p86-N4096-x30-xa28-fr008-g3-dt80-w10_N_50000.dat",
            #     "Name": "fsi-T220-dt80-xa28-w10_N_50000.dat",
            #     "Temperature": "220",
            #     "tc_p0": 50,
            #     "tmin_int": 50,
            #     "tmax_int": 60,
            #     "tm_ajuste":57,
            #     "t_max_exp": 80,
            #     "output": "/home/elena/fit/H-Pt/ajustes/fits-T220-k0p86-N4096-x30-xa28-fr008-g3-dt80-w10_N_50000.pdf"
            # },
            {
                "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T220-k0p86-N4096-x30-xa28-fr008-g3-dt80-w10_N_80000.dat",
                "Name": "fsi-T220-dt80-xa28-w10_N_80000.dat",
                "Temperature": "220",
                "tc_p0": 20,
                "tmin_int": 20,
                "tmax_int": 60,
                "tm_ajuste":61,
                "t_max_exp": 80,
                "output": "/home/elena/fit/H-Pt/ajustes/fits-T220-k0p86-N4096-x30-xa28-fr008-g3-dt80-w10_N_80000.pdf"
            },
            # {
            #     "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T220-k0p86-N4096-x30-xa28-fr008-g3-dt80-w8_N_50000.dat",
            #     "Name": "fsi-T220-dt80-xa28-w8_N_50000.dat",
            #     "Temperature": "220",
            #     "tc_p0": 20,
            #     "tmin_int": 20,
            #     "tmax_int": 60,
            #     "tm_ajuste":61,
            #     "t_max_exp": 90,
            #     "output": "/home/elena/fit/H-Pt/ajustes/fits-T220-k0p86-N4096-x30-xa28-fr008-g3-dt80-w8_N_50000.pdf"
            # },
            # {
            #     "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T220-k0p86-N4096-x32-xa30-fr008-g3-dt80-w10_N_20000.dat",
            #     "Name": "fsi-T220-dt80-xa30-w10_N_20000.dat",
            #     "Temperature": "220",
            #     "tc_p0": 20,
            #     "tmin_int": 20,
            #     "tmax_int": 60,
            #     "tm_ajuste":45,
            #     "t_max_exp": 80,
            #     "output": "/home/elena/fit/H-Pt/ajustes/fits-T220-k0p86-N4096-x32-xa30-fr008-g3-dt80-w10_N_20000.pdf"
            # }
            # ,
            # {
            #     "file": "/home/elena/fit/H-Pt/T-bajas/fsi-T220-k0p86-N4096-x32-xa30-fr008-g3-dt80-w10_N_20000.dat",
            #     "Name": "fsi-T220-dt80-xa30-t90_N_20000.dat",
            #     "Temperature": "220",
            #     "tc_p0": 20,
            #     "tmin_int": 20,
            #     "tmax_int": 60,
            #     "tm_ajuste":45,
            #     "t_max_exp": 100,
            #     "output": "/home/elena/fit/H-Pt/ajustes/fits-T220-k0p86-N4096-x32-xa30-fr008-g3-dt80-w10_N_20000.pdf"
            # }
        ]

colors = ['b', 'orange', 'g', 'k', 'm', 'c', 'y' ]
# Función principal para procesar múltiples datasets
def procesar_multiples_datasets():
    # Inicializar listas para acumular valores de b, R2 y tiempos_minimos
    b_values_all = []
    R2_values_all = []
    tiempos_minimos_all = []
    colors = ['b', 'r', 'g', 'k']
    # Definir las constantes k y a
    k = 0.86
    a = 2.77

    # Calcular el factor 1 - cos(k * a)
    factor = 1 - np.cos(k*a*np.cos(np.pi/6))

    # Registrar el tiempo de inicio
    start_time_total = time.time()

    # Abrir archivo para guardar los resultados
    with open('decay-H-Elena-fr008.dat', 'w') as fileID:
        for dataset_index, data_info in enumerate(datasets):  # Adding dataset_index
            try:
                start_time = time.time()
                print(f"Inicio del procesamiento del archivo: {data_info['file']}")

                # Cargar los datos desde el archivo especificado
                try:
                    data = np.loadtxt(data_info['file'])
                    t = data[:, 0]  # Tiempo
                    y = data[:, 1]  # Señal
                except Exception as e:
                    raise RuntimeError(f"Error cargando datos desde {data_info['file']}: {e}")

                # Filtrar tiempos menores a t_max_exp
                try:
                    start_time_filter = time.time()
                    idx_corte = t < data_info['t_max_exp']
                    t_filtrado = t[idx_corte]
                    y_filtrado = y[idx_corte]
                    if not t_filtrado.size or not y_filtrado.size:
                        raise ValueError("Datos vacíos después del filtrado.")
                except Exception as e:
                    raise RuntimeError(f"Error filtrando datos en {data_info['file']}: {e}")

                if not t_filtrado.size or not y_filtrado.size:
                    print(f"Archivo {data_info['file']} no tiene datos válidos después del filtrado.")
                    continue  # Salta al siguiente archivo en datasets

                # Ajuste exponencial inicial
                A, b, C = ajustar_exponencial_auto_p0(t_filtrado, y_filtrado, data_info['tc_p0'])
                p0 = [A, b, C]

                # Usar el nombre de salida proporcionado en data_info
                with PdfPages(data_info['output']) as pdf:
                    # Graficar y guardar los parámetros, pasando dataset_index
                    b_ext, R2_ext, tiempos_minimos, b_values, R2_values, b_final = graficar_parametros_tc_known(
                        t_filtrado, y_filtrado, t, y, data_info['tmin_int'], data_info['tmax_int'],
                        p0, data_info['tm_ajuste'], data_info['t_max_exp'], pdf, data_info, dataset_index  # Added here
                    )

                    # Aplicar el factor (1 - cos(k * a)) al valor de b_final
                    b_final_adjusted = b_final / factor

                    # Acumular valores de b y R2
                    b_values_all.append(b_values)
                    R2_values_all.append(R2_values)
                    tiempos_minimos_all.append(tiempos_minimos)

                # Escribir el valor ajustado de b_final en el archivo de salida
                fileID.write(f"{data_info['Temperature']} {b_final_adjusted}\n")

            except Exception as e:
                print(f"Error procesando el archivo {data_info['file']}: {str(e)}")

    # Llamar a la función de graficado de b y R^2
    start_time_plot_all = time.time()
    output_pdf = 'b_R2_vs_tmin_plots.pdf'  # Set output file for the plots
    graficar_b_R2_vs_tmin(tiempos_minimos_all, b_values_all, R2_values_all, datasets, 'r^2.pdf')
    graficar_ajustes_exponenciales(datasets, 'I-vs-b.pdf')
    print(f"Gráficos combinados generados y guardados en {time.time() - start_time_plot_all:.2f} segundos")
    print(f"Tiempo total de procesamiento: {time.time() - start_time_total:.2f} segundos")

# Función para ajuste exponencial
def ajustar_exponencial_auto_p0(t, y, tc_p0):
    def modelo_exponencial(t, A, b, C):
        return A * np.exp(-b * t) + C

    # Check if the data is empty
    if len(y) == 0:
        raise ValueError("El conjunto de datos 'y' está vacío.")
    
    if len(y) > 0:
        A0 = max(y)
        C0 = min(y)
    else:
        A0 = 1e-6
        C0 = 1e-6

    if C0 <= 0 or A0 <= 0:
        C0 = 1e-6  # Small positive value to avoid log(0)
        A0 = max(y) if max(y) > 0 else 1e-6  # Ensure A0 is positive

    if (A0 > 0 and C0 > 0):
        b0 = np.log(A0 / C0) / (max(t) - min(t))
    else:
        b0 = 1.0  # Fallback value for b0
    
    try:
        popt, _ = curve_fit(modelo_exponencial, t, y, p0=[A0, b0, C0], bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=100000, method='dogbox')
    except RuntimeError as e:
        print(f"Curve fit failed: {e}")
        popt = [A0, b0, C0]  # Fallback to initial guesses if curve fit fails

    A, b, C = popt
    return A, b, C

# Función para graficar parámetros
def graficar_parametros_tc_known(t, y, t_data, y_data, tmin_int,  tmax_int, p0, tm_ajuste, t_max_exp, pdf, data_info, dataset_index):
    tiempos_minimos = np.linspace(tmin_int, tmax_int, 200)
    b_values = []
    R2_values = []

    for tmin in tiempos_minimos:
        idx = t > tmin
        t_filtrado = t[idx]
        y_filtrado = y[idx]

        if len(t_filtrado) > 1:
            A, b, C = ajustar_exponencial_auto_p0(t_filtrado, y_filtrado, p0[1])
            y_ajustado = A * np.exp(-b * t_filtrado) + C
            ss_res = np.sum((y_filtrado - y_ajustado) ** 2)
            ss_tot = np.sum((y_filtrado - np.mean(y_filtrado)) ** 2)
            R2 = 1 - (ss_res / ss_tot)
            b_values.append(b)
            R2_values.append(R2)

    tiempos_minimos_filtered = [t for t, r2 in zip(tiempos_minimos, R2_values) if r2 > 0.30]
    b_values_filtered = [b for b, r2 in zip(b_values, R2_values) if r2 > 0.30]
    R2_values_filtered = [r2 for r2 in R2_values if r2 > 0.30]
    
    b_ext = max(b_values_filtered, key=lambda b: b)
    R2_ext = max(R2_values_filtered, key=lambda r: r)
    
    idx_tm = np.abs(tiempos_minimos - tm_ajuste).argmin()
    if idx_tm < len(R2_values):
        R2_tm_ajuste = R2_values[idx_tm]
    else:
        R2_tm_ajuste = None

    plt.figure(figsize=(10, 8))
    A, b, C = ajustar_exponencial_auto_p0(t[t > tm_ajuste], y[t > tm_ajuste], p0[1])
    b_final= b

    color = colors[dataset_index % len(colors)]  # Seleccionar color basado en dataset_index

    plt.subplot(2, 2, 1)
    plt.plot(t_data, y_data, 'o', color=color, label="NS", zorder=5)
    #plt.plot(t_data, A * np.exp(-b * t_data) + C, '-', color='r', label=f'Ajuste (b={b:.4f}, C={C:.4f}, R²={R2_tm_ajuste:.4f})')
    plt.plot(t_data, A * np.exp(-b * t_data) + C, '-', color='r', label=rf"$A e^{{-\alpha t}} + C, \quad \alpha = {b:.4f} ps^{-1}$", zorder=5)
    plt.xlabel(r't [ps]')
    plt.ylabel(r'$I(\Delta k, t)$')
        # Ensure legend is always on top
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=1, frameon=True)

    # Manually set the zorder of the legend
    legend.set_zorder(10)  # Moves legend above all other elements
    plt.xscale('log')
    plt.text(0.4, 0.15, f"T = {data_info['Temperature']}K", transform=plt.gca().transAxes, fontsize=12, 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.xlim(10**1, t_max_exp+10)
    plt.ylim(min(y_data), max(y_data))
    plt.grid(True, which='both', linestyle='--')

    # Define filename for saving
    first_subplot_filename = f"exponential_fit_T{data_info['Temperature']}-fr008.pdf"

    # Save only the first subplot figure as a separate file with tight bounding box
    plt.savefig(first_subplot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f"Saved first subplot as {first_subplot_filename}")

    # Close the figure after saving it to avoid overlap with other subplots
    plt.close()

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t_data, y_data, 'o', color=color, label="NS", zorder=3)
    #plt.plot(t_data, A * np.exp(-b * t_data) + C, '-', color='r', label=f'Ajuste (b={b:.4f}, C={C:.4f}, R²={R2_tm_ajuste:.4f})')
    plt.plot(t_data, A * np.exp(-b * t_data) + C, '-', color='r', label=rf"$A e^{{-\alpha t}} + C, \quad \alpha = {b:.4f} ps^{-1}$", zorder=3)
    plt.xlabel(r't [ps]')
    plt.ylabel(r'$I(\Delta k, t)$')
    # Ensure legend is always on top
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=1, frameon=True)

    # Manually set the zorder of the legend
    legend.set_zorder(10)  # Moves legend above all other elements

    plt.xscale('log')
    plt.text(0.4, 0.15, f"T = {data_info['Temperature']}K", transform=plt.gca().transAxes, fontsize=12, 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.xlim(10**1, t_max_exp+10)
    plt.ylim(min(y_data), max(y_data))
    plt.grid(True, which='both', linestyle='--')


    plt.subplot(2, 2, 2)
    plt.plot(tiempos_minimos_filtered, R2_values_filtered, color=color, label='R²')
    plt.xlabel('Tiempo mínimo para ajuste [ps]')
    plt.ylabel('Valor de R²')
    plt.title('Cambio del valor de R² con el tiempo mínimo')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.grid()
    plt.legend()
    plt.ylim(min(R2_values_filtered), max(R2_values_filtered))
    plt.gca().invert_xaxis()

    plt.subplot(2, 2, 3)
    plt.plot(tiempos_minimos_filtered, b_values_filtered, color=color, label='b')
    plt.xlabel('Tiempo mínimo para ajuste [ps]')
    plt.ylabel('Valor de b')
    plt.title('Cambio del valor de b con el tiempo mínimo')
    plt.grid()
    plt.ylim(min(b_values_filtered), max(b_values_filtered))
    plt.legend()
    plt.gca().invert_xaxis()

    plt.subplot(2, 2, 4)
    plt.plot(t[t > tm_ajuste], y[t > tm_ajuste], 'o', color=color, label='Datos (t > tmin)')
    plt.plot(t[t > tm_ajuste], A * np.exp(-b * t[t > tm_ajuste]) + C, '-', color='r', label='Ajuste Exponencial')
    plt.xlabel('Tiempo [ps]')
    plt.ylabel('Señal')
    plt.title(f'Ajuste Exponencial desde t > {tm_ajuste}')
    plt.legend()
    plt.grid()
    plt.subplots_adjust(hspace=0.4)

    pdf.savefig()
    plt.close()
    
    return b_ext, R2_ext, tiempos_minimos, b_values, R2_values, b_final

def graficar_b_R2_vs_tmin(tiempos_minimos_all, b_values_all, R2_values_all, datasets, output_pdf):
    # Prepare lists for the values corresponding to the maximum R²
    b_values_max_r2 = []
    R2_max_values = []
    tmin_values_max_r2 = []

    # Extract the values corresponding to the maximum R² for each dataset
    for tiempos_minimos, b_values, R2_values in zip(tiempos_minimos_all, b_values_all, R2_values_all):
        max_r2_idx = np.argmax(R2_values)
        b_values_max_r2.append(b_values[max_r2_idx])
        R2_max_values.append(R2_values[max_r2_idx])
        tmin_values_max_r2.append(tiempos_minimos[max_r2_idx])

    # Create the PDF to save the figure with plots and table on the same page
    with PdfPages(output_pdf) as pdf:
        # Define a GridSpec with 2 rows and 1 column; the table will span across the bottom
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, height_ratios=[4, 4, 0.2])  # Adjust height_ratios to reduce space for the table

        # Subplot 1: b vs tiempo mínimo
        ax1 = fig.add_subplot(gs[0, 0])
        dark_yellow = (0.55, 0.55, 0.0)
        light_blue = (0.4, 0.7, 1.0) 
        lines1 = []
        colors = ['purple','green', light_blue, 'orange', dark_yellow,'blue', 'red','black', 'brown', 'pink', 'magenta', 'blue', 'gray', 'olive', 'lime', 'teal', 'navy', 'maroon', 'aqua', 'fuchsia', 'silver', 'lime', 'teal', 'navy', 'maroon', 'aqua', 'fuchsia', 'silver']
        for (tiempos_minimos, b_values, R2_values, data_info), color in zip(zip(tiempos_minimos_all, b_values_all, R2_values_all, datasets), colors):
            # Filter the data to include only points where R² > 0.50
            tiempos_minimos_filtered = [t for t, r2 in zip(tiempos_minimos, R2_values) if r2 > 0.30]
            b_values_filtered = [b for b, r2 in zip(b_values, R2_values) if r2 > 0.30]
            line, = ax1.plot(tiempos_minimos_filtered, b_values_filtered, label=f"{data_info['file']}", linewidth=3, color=color)
            lines1.append(line)
        ax1.set_xlabel('Tiempo mínimo [ps]', fontsize=14)
        ax1.set_ylabel('Valor de b', fontsize=14)
        ax1.set_title('b vs Tiempo mínimo', fontsize=14)
        #ax1.legend(fontsize=10)
        ax1.grid()
        ax1.invert_xaxis()

        # Subplot 2: R² vs tiempo mínimo
        ax2 = fig.add_subplot(gs[0, 1])
        lines2 = []
        for (tiempos_minimos, R2_values, data_info), color in zip(zip(tiempos_minimos_all, R2_values_all, datasets), colors):
            # Filter the data to include only points where R² > 0.30
            tiempos_minimos_filtered = [t for t, r2 in zip(tiempos_minimos, R2_values) if r2 > 0.30]
            R2_values_filtered = [r2 for r2 in R2_values if r2 > 0.30]
            line, = ax2.plot(tiempos_minimos_filtered, R2_values_filtered, label=f"{data_info['file']}", linewidth=3, color=color)
            lines2.append(line)
        ax2.set_xlabel('Tiempo mínimo [ps]', fontsize=14)
        ax2.set_ylabel('Valor de R²', fontsize=14)
        ax2.set_title('R² vs Tiempo mínimo', fontsize=14)
        
        #ax2.legend(fontsize=10)
        ax2.grid()
        ax2.invert_xaxis()

        # Table: Display maximum R² values and corresponding tiempos mínimos and b values
        ax_table = fig.add_subplot(gs[1:, :])  # Span the table across the full width
        ax_table.axis('tight')
        ax_table.axis('off')

        # Prepare the data for the table
        table_data = [
            [
                data_info['Name'],
                f"{b:.4f}",
                f"{r2:.4f}",
                f"{tmin:.2f}"
            ]
            for data_info, b, r2, tmin in zip(datasets, b_values_max_r2, R2_max_values, tmin_values_max_r2)
        ]
        col_labels = ['Dataset', 'Valor de b', 'R² (Max)', 'tmin (Max R²)']

        # Calculate column widths based on the maximum length of the text in each column
        col_widths = [max(len(str(item)) for item in col) for col in zip(*table_data, col_labels)]
        col_widths = [width*0.1 for width in col_widths]  # Adjust the scaling factor as needed

        # Create the table
        table = ax_table.table(
            cellText=table_data, 
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.08, 0.08, 0.08] 
        )
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.5, 1.5)  # Adjust the size of the table

        # Apply colors to the dataset names in the table
        for i, line in enumerate(lines1):
            table[(i+1, 0)].set_text_props(color=line.get_color())

        # Adjust layout to reduce space between subplots and table
        fig.subplots_adjust(hspace=0.05, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.1)
        # Save the figure with both plots and table on the same page
        pdf.savefig(fig)  # Save the combined figure to the PDF
        #plt.show() 
        plt.close(fig)    # Close the figure to free memory  
        if plt.get_fignums():  # Comprueba si hay figuras creadas
            pdf.savefig()
        else:
            print(f"No se generaron figuras para {data_info['file']}. No se guardará PDF.")
    print(f"Combined figure with plots and table saved to {output_pdf}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

def graficar_ajustes_exponenciales(datasets, output_pdf):
    """
    Genera una figura con múltiples subgráficos, cada uno correspondiente a un dataset,
    mostrando el ajuste exponencial o `tm_ajuste` como detalle.
    """
    num_datasets = len(datasets)
    num_cols = 3  # Número de columnas en la figura
    num_rows = int(np.ceil(num_datasets / num_cols))  # Determinar filas necesarias
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Asegurar un solo índice para los ejes
    # Definir colores personalizados
    dark_yellow = (0.55, 0.55, 0.0)
    light_blue = (0.4, 0.7, 1.0)
    colors = ['purple', 'green', light_blue, 'orange', dark_yellow, 'blue', 'red', 'black', 
              'brown', 'pink', 'magenta', 'blue', 'gray', 'olive', 'lime', 'teal', 'navy', 
              'maroon', 'aqua', 'fuchsia', 'silver', 'lime', 'teal', 'navy', 'maroon', 
              'aqua', 'fuchsia', 'silver']
    
    for i, data_info in enumerate(datasets):
        try:
            data = np.loadtxt(data_info['file'])
            t = data[:, 0]  # Tiempo
            y = data[:, 1]  # Señal
        except Exception as e:
            print(f"Error cargando datos desde {data_info['file']}: {e}")
            continue  # Pasar al siguiente dataset si hay error

        tm_ajuste = data_info.get('tm_ajuste', None)  # Obtener tm_ajuste si existe

        # Filtrado de datos
        if tm_ajuste is not None:
            mask = (t > tm_ajuste) & (t < data_info['t_max_exp'])
            t_filtrado_min = t[mask]
            y_filtrado_min = y[mask]
        else:
            t_filtrado_min = t
            y_filtrado_min = y

        ax = axes[i]
        color_idx = i % len(colors)  # Para ciclar por la lista de colores
        ax.plot(t_filtrado_min, y_filtrado_min, 'o', label='Datos', color=colors[color_idx])

        if tm_ajuste is not None:
            ax.axvline(tm_ajuste, color='r', linestyle='--', label=f'tm_ajuste: {tm_ajuste:.2f}')

        # Ajuste exponencial
        def modelo_exponencial(t, A, b, C):
            return A * np.exp(-b * t) + C
        
        if len(y_filtrado_min) > 0:
            A0 = max(y_filtrado_min)
            C0 = min(y_filtrado_min)
        else:
            A0 = 1e-6
            C0 = 1e-6

        if C0 <= 0 or A0 <= 0:
            C0 = 1e-6  # Evitar log(0)
            A0 = max(y_filtrado_min) if max(y_filtrado_min) > 0 else 1e-6

        if len(t_filtrado_min) > 1 and (A0 > 0 and C0 > 0):
            b0 = np.log(A0 / C0) / (max(t_filtrado_min) - min(t_filtrado_min))
        else:
            b0 = 1.0  # Valor de respaldo para b0
        
        try:
            popt, _ = curve_fit(modelo_exponencial, t_filtrado_min, y_filtrado_min, p0=[A0, b0, C0], bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=100000, method='dogbox')
            x_fit = np.linspace(min(t_filtrado_min), max(t_filtrado_min), 200)
            y_fit = modelo_exponencial(x_fit, *popt)
            ax.plot(x_fit, y_fit, label=f'Ajuste Exp: b={popt[1]:.4f}, C={popt[2]:.4f}', color=colors[(color_idx + 1) % len(colors)], linewidth=4)
        except Exception as e:
            print(f"Error en el ajuste para {data_info['file']}: {e}")

        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Valor')
        ax.set_title(data_info.get('Name', 'Dataset'))
        ax.legend()
        ax.grid()
        #plt.show()
    
    # Eliminar subgráficos vacíos si hay menos datasets que subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Guardar en PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Figura guardada en {output_pdf}")

procesar_multiples_datasets()

