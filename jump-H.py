import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv  # Bessel function of the first kind
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle


# Set font family globally
plt.rcParams['font.family'] = 'serif'  # or any other font family you prefer


# Define the gamma values
gamma1 = 1.364
gamma2 = 0.126
gamma3 = 0.0154

# Define the functions for P0H, P1H, and P2H
def P0H1(t): return iv(0, gamma1 * t) * np.exp(-gamma1 * t)
def P0H2(t): return iv(0, gamma2 * t) * np.exp(-gamma2 * t)
def P0H3(t): return iv(0, gamma3 * t) * np.exp(-gamma3 * t)

def P1H(t): return iv(1, gamma1 * t) * np.exp(-gamma1 * t)
def P2H(t): return iv(1, gamma2 * t) * np.exp(-gamma2 * t)
def P3H(t): return iv(1, gamma3 * t) * np.exp(-gamma3 * t)

def P1H2(t): return iv(2, gamma1 * t) * np.exp(-gamma1 * t)
def P2H2(t): return iv(2, gamma2 * t) * np.exp(-gamma2 * t)
def P3H2(t): return iv(2, gamma3 * t) * np.exp(-gamma3 * t)

# Time range
t_max = 500
t = np.linspace(0, t_max, 500)

# Create the plot for GraphP1H
fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(t, P0H1(t), label=r'$P_0(t)$', color='blue', linewidth=4)
ax.plot(t, P1H(t), label=r'$P_1(t)$', color='black', linestyle=':', linewidth=4)
ax.plot(t, P1H2(t), label=r'$P_2(t)$', color='red', linestyle='--', linewidth=4)

t_max1 = 150
# Customizing the plot
ax.set_xlim([-0.01, t_max1])
ax.set_ylim([0, 0.50])
ax.set_xlabel(r'$t [ps]$', fontsize=18, fontfamily='serif')
ax.set_ylabel(r'$P_n(t)$', fontsize=18, fontfamily="serif")
ax.legend(loc='upper right', fontsize=14)

# Set specific ticks on x-axis
ax.set_xticks([0, 50, 100, 150])

# Adding a text annotation
ax.text(50, 0.45, 'T=214K', fontsize=18, family='serif')
ax.tick_params(axis='both', which='major', labelsize=16)

# Adjust layout to avoid clipping ylabel
fig.tight_layout()

# Save figure as PDF
plt.savefig("Jump-T214K.pdf", dpi=300, bbox_inches='tight')
plt.close()

# Plot P0H2, P2H
fig2, ax2 = plt.subplots(figsize=(5, 5))

ax2.plot(t, P0H2(t), label=r'$P_0(t)$', color='blue', linewidth=4)
ax2.plot(t, P2H(t), label=r'$P_1(t)$', color='black', linestyle=':', linewidth=4)
ax2.plot(t, P2H2(t), label=r'$P_2(t)$', color='red', linestyle='--', linewidth=4)

t_max2 = 150
# Customizing the plot
ax2.set_xlim([-0.01, t_max2])
ax2.set_ylim([0, 0.5])
ax2.set_ylabel(r'$P_n(t)$', fontsize=18, fontfamily='serif')
ax2.set_xlabel(r'$t [ps]$', fontsize=18, fontfamily='serif')
#ax2.legend(loc='upper right', fontsize=14)

# Adding a text annotation
ax2.text(50, 0.45, 'T=121K', fontsize=18, family='serif')
ax2.tick_params(axis='both', which='major', labelsize=16)
# Adjust layout to avoid clipping ylabel
fig.tight_layout()

# Save figure as PDF
plt.savefig("Jump-T121K.pdf", dpi=300, bbox_inches='tight')
plt.close()


# Plot P0H3, P3H
fig3, ax3 = plt.subplots(figsize=(5, 5))

ax3.plot(t, P0H3(t), label=r'$P_0(t)$', color='blue', linewidth=4)
ax3.plot(t, P3H(t), label=r'$P_1(t)$', color='black', linestyle=':', linewidth=4)
ax3.plot(t, P3H2(t), label=r'$P_2(t)$', color='red', linestyle='--', linewidth=4)

# Customizing the plot
ax3.set_xlim([-0.01, t_max])
ax3.set_ylim([0, 0.5])
ax3.set_ylabel(r'$P_n(t)$', fontsize=18, fontfamily='serif')
ax3.set_xlabel(r'$t [ps]$', fontsize=18, fontfamily="serif")
#ax3.legend(loc='upper right', fontsize=14)

# Adding a text annotation
ax3.text(160, 0.45, 'T=80K', fontsize=18, family='serif')
ax3.tick_params(axis='both', which='major', labelsize=16)

# Create the inset for the zoomed-in plot using bbox_to_anchor and normalized coordinates
# Specify the position of the inset axes using bbox_to_anchor
axins = inset_axes(ax3, width=1.1, height=1.1, loc='upper right', borderpad=2)
axins.plot(t, P0H3(t), label=r'$P_0(t)$', color='blue', linewidth=4)
axins.plot(t, P3H(t), label=r'$P_1(t)$', color='black', linestyle=':', linewidth=4)
axins.plot(t, P3H2(t), label=r'$P_2(t)$', color='red', linestyle='--', linewidth=4)
#plt.show()
# Set the limits for the inset zoom
axins.set_xlim([0, 100])  # Zooming into the first 100 units on the x-axis
axins.set_ylim([0, 0.5])
axins.set_facecolor('white')  # Set background color
axins.patch.set_alpha(0.7)    # Set transparency
axins.spines['top'].set_visible(True)  # Show borders
axins.spines['right'].set_visible(True)
# Adjust layout to avoid clipping ylabel
#fig3.tight_layout()

# Save figure as PDF
plt.savefig("Jump-T80K.pdf", dpi=300, bbox_inches='tight')
plt.close()
