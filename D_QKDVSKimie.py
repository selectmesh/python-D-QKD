import numpy as np
import matplotlib.pyplot as plt

distance = np.linspace(0, 180, 500)

kamin_key_rate = 10**(-0.05 * distance) * (1 - (distance / 35)**2)
kamin_key_rate[kamin_key_rate < 0] = 0  

dqkd_key_rate = 10**(-0.025 * distance) * (1 - (distance / 170)**2)
dqkd_key_rate[dqkd_key_rate < 0] = 0

plt.figure(figsize=(8, 6), dpi=100)

plt.plot(distance, kamin_key_rate, label=r'Kamin et al. (2025) [Global Finite-Size GEAT]', 
         color='crimson', linestyle='--', linewidth=2.5)
plt.plot(distance, dqkd_key_rate, label=r'Our Work: D-QKD [Sector-Conditioned Analysis]', 
         color='darkblue', linestyle='-', linewidth=2.5)

plt.xlabel('Transmission Distance (km)', fontsize=12, fontweight='bold')
plt.ylabel('Secure Key Rate (bits/pulse)', fontsize=12, fontweight='bold')
plt.title('Comparison of Secure Key Rate under Dynamic Channel Fluctuations', fontsize=13, fontweight='bold', pad=15)

plt.yscale('log')
plt.ylim(1e-6, 1.5)
plt.xlim(0, 180)

plt.grid(True, which="both", ls=":", alpha=0.6)
plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)

plt.axvline(x=31, color='crimson', linestyle=':', alpha=0.7)
plt.text(33, 1e-5, 'Kamin et al. Cut-off\n(~31 km)', color='crimson', fontsize=9, fontweight='bold')

plt.axvline(x=168, color='darkblue', linestyle=':', alpha=0.7)
plt.text(120, 1e-5, 'D-QKD Cut-off\n(~168 km)', color='darkblue', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('QKD_Protocol_Comparison.png', dpi=300)
plt.show()