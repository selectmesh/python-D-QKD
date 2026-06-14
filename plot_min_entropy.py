# python plot_min_entropy.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9.5, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

def h2(x):
    x = np.clip(x, 1e-15, 1-1e-15)
    return -x*np.log2(x)-(1-x)*np.log2(1-x)

alpha=0.2; eta_d=0.15; p_dk=1e-6; fEC=1.16; qs=0.5; M=8; mu=0.6; N=1e12

def eta(L): return 10**(-alpha*L/10)
def Q1(L):  return 1-(1-eta_d*eta(L))*(1-p_dk)

# Sector sensitivity: some sectors very sensitive to distance
sens = np.array([0.05, 0.10, 0.35, 0.85, 1.0, 0.80, 0.30, 0.08])

distances = np.linspace(0, 200, 2000)
Hmin_std = np.zeros_like(distances)
Hmin_sc  = np.zeros_like(distances)

for i, L in enumerate(distances):
    Q = Q1(L)
    e0 = 0.5*p_dk/max(Q,1e-15)
    
    # Sector errors: good sectors stay clean, bad sectors blow up
    e_sec = np.clip(e0 + 0.015 + sens*0.035*(L/8)**1.1, 0.001, 0.499)
    pm = np.ones(M)/M
    e_gl = np.sum(pm*e_sec)
    
    # Std: global
    Rp = qs*Q*mu*(1-h2(e_gl))
    Rc = fEC*qs*Q*mu*h2(e_gl)
    fs = 5*np.sqrt(np.log2(1/1e-10)/N)*qs*Q*mu
    R_std = Rp-Rc-fs
    Hmin_std[i] = max(N*(Rp-fs),0) if R_std>0 else 0
    
    # SC: per-sector + selection
    Rp_s = qs*Q*mu*(1-h2(e_sec))
    Rc_s = fEC*qs*Q*mu*h2(e_sec)
    fs_s = 5*np.sqrt(M*np.log2(1/1e-10)/N)*qs*Q*mu
    R_s = Rp_s-Rc_s-fs_s
    kept = R_s>0
    if np.any(kept):
        Rt = np.sum(pm[kept]*R_s[kept])
        Hmin_sc[i] = max(N*np.sum(pm[kept]*Rp_s[kept]),0) if Rt>0 else 0

d_std = distances[Hmin_std>0][-1] if np.any(Hmin_std>0) else 0
d_sc  = distances[Hmin_sc>0][-1]  if np.any(Hmin_sc>0)  else 0
print(f"Std: {d_std:.0f} km | SC: {d_sc:.0f} km | x{d_sc/max(d_std,1):.1f}")

# Panel (b)
L_fix=25; Q_f=Q1(L_fix)
e_bf = 0.5*p_dk/max(Q_f,1e-15)+0.025
A_vals = np.linspace(0,0.12,400)
var_l=[]; gain_l=[]; bnd_l=[]
for A in A_vals:
    pm=np.ones(M)/M
    es=np.clip(e_bf+A*np.cos(2*np.pi*np.arange(M)/M),0.001,0.499)
    ea=np.sum(pm*es); ve=np.sum(pm*(es-ea)**2)
    var_l.append(ve)
    hg=qs*Q_f*mu*(1-h2(ea))
    hs=qs*Q_f*mu*np.sum(pm*(1-h2(es)))
    gain_l.append(max(N*(hs-hg),0))
    bnd_l.append(N*qs*Q_f*mu*(2/np.log(2))*ve)
va=np.array(var_l); ga=np.array(gain_l); ba=np.array(bnd_l)

# FIGURE
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14.5,5.8))

ms=Hmin_sc>0; mg=Hmin_std>0
ax1.semilogy(distances[ms],Hmin_sc[ms],color='#2471A3',lw=2.8,zorder=3,
             label=r'$H_{\min}^{\varepsilon,\,\mathrm{sc}}$  (This work)')
ax1.semilogy(distances[mg],Hmin_std[mg],color='#C0392B',lw=2.8,ls='--',zorder=3,
             label=r'$H_{\min}^{\varepsilon,\,\mathrm{std}}$  (Standard GEAT)')

both=mg&ms&(Hmin_sc>Hmin_std)
ax1.fill_between(distances[both],Hmin_std[both],Hmin_sc[both],
                 alpha=0.20,color='#27AE60',label=r'$\delta H_{\min}$ gain')
ext=ms&~mg
if np.any(ext):
    ax1.fill_between(distances[ext],10,Hmin_sc[ext],
                     alpha=0.12,color='#2471A3',label='SC-only zone')

ax1.axvline(d_std,color='#C0392B',ls=':',alpha=0.6,lw=1.3)
ax1.axvline(d_sc, color='#2471A3',ls=':',alpha=0.6,lw=1.3)

yt=2e2
ax1.annotate(f'Std aborts\n({d_std:.0f} km)',xy=(d_std,yt),fontsize=9.5,
             color='#C0392B',fontweight='bold',ha='right',
             xytext=(-12,22),textcoords='offset points',
             arrowprops=dict(arrowstyle='->',color='#C0392B',lw=1.3))
ax1.annotate(f'SC extends\n({d_sc:.0f} km)',xy=(d_sc,yt),fontsize=9.5,
             color='#2471A3',fontweight='bold',ha='left',
             xytext=(8,22),textcoords='offset points',
             arrowprops=dict(arrowstyle='->',color='#2471A3',lw=1.3))

if d_sc>d_std:
    ym=40
    ax1.annotate('',xy=(d_sc,ym),xytext=(d_std,ym),
                 arrowprops=dict(arrowstyle='<->',color='#27AE60',lw=2.2))
    ax1.text((d_std+d_sc)/2,ym*5,
             f'$\\times${d_sc/max(d_std,1):.1f} distance gain',
             ha='center',fontsize=11,color='#27AE60',fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#27AE60',alpha=0.85))

ax1.set_xlabel('Distance (km)')
ax1.set_ylabel(r'Smooth min-entropy  $H_{\min}^{\varepsilon}(S^N | E\,I^N)$  [bits]')
ax1.set_title(r'(a)  Min-entropy: sector-conditioned vs. standard  ($N\!=\!10^{12}$)')
ax1.set_xlim(0,200); ax1.set_ylim(10,5e9)
ax1.legend(loc='upper right',framealpha=0.92,edgecolor='gray')
ax1.grid(True,alpha=0.25,which='both')

ax2.plot(va*1e4,ga,color='#2471A3',lw=2.5,label=r'Actual $\delta H_{\min}$')
ax2.plot(va*1e4,ba,color='#E67E22',lw=2.2,ls='--',
         label=r'Thm. bound: $N q_{\mathrm{sift}} \bar{Q}_1 \frac{2}{\ln 2}\mathrm{Var}_m$')
ax2.fill_between(va*1e4,ba,ga,where=(ga>=ba),alpha=0.12,color='#2471A3',
                 label='Higher-order concavity')

Ap=0.025; vp=Ap**2/2; ip=np.argmin(np.abs(va-vp))
ax2.plot(vp*1e4,ga[ip],'o',color='#C0392B',ms=10,zorder=5,
         markeredgecolor='white',markeredgewidth=1.5)
ax2.annotate(f'Benchmark\n$\\delta H_{{\\min}}\\!\\approx\\!{ga[ip]:.2e}$',
             xy=(vp*1e4,ga[ip]),fontsize=9,color='#C0392B',fontweight='bold',
             xytext=(22,-18),textcoords='offset points',
             arrowprops=dict(arrowstyle='->',color='#C0392B',lw=1.2))

ax2.set_xlabel(r'Sector variance  $\mathrm{Var}_m(e_{1,\mathrm{ph}}^{(m)})$  [$\times 10^{-4}$]')
ax2.set_ylabel(r'Min-entropy gain  $\delta H_{\min}$  [bits]')
ax2.set_title(r'(b)  $\delta H_{\min}$ vs. sector variance  (25 km, $N\!=\!10^{12}$)')
ax2.legend(loc='upper left',framealpha=0.92,edgecolor='gray')
ax2.grid(True,alpha=0.25)
ax2.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))

plt.tight_layout(w_pad=3)
plt.savefig('min_entropy_comparison.pdf',format='pdf')
plt.savefig('min_entropy_comparison.png',format='png')
print("Done!")