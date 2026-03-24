import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/VitalQuest.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.date
df.columns = df.columns.str.strip()

# Rename for readability
df = df.rename(columns={
    'Protein_g': 'Protein',
    'Carbs_g': 'Carbs',
    'Fat_g': 'Fat',
    'Micronutrient_Score': 'Micronutrient Score',
    'Sleep_Hours': 'Sleep Hours',
    'Workout_Intensity': 'Workout Intensity',
    'Energy_Level': 'Energy Level',
    'Recovery_Score': 'Recovery Score'
})

numeric_cols = ['Calories', 'Protein', 'Carbs', 'Fat',
                'Micronutrient Score', 'Sleep Hours', 'Steps',
                'Workout Intensity', 'Energy Level', 'Recovery Score']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# ── 2. Styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': '#F8FAFC',
    'axes.facecolor': '#F8FAFC',
})

BRAND   = '#2563EB'   # blue
ACCENT  = '#16A34A'   # green
WARN    = '#D97706'   # amber
DARK    = '#1E293B'   # slate dark
GRAY    = '#64748B'   # slate gray

# ── 3. Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22), facecolor='#F8FAFC')
fig.suptitle('VitalQuest – Python Analytics Report',
             fontsize=22, fontweight='bold', color=DARK, y=0.98)
fig.text(0.5, 0.965, '90-Day Wellness Dataset  |  Correlation & Regression Analysis',
         ha='center', fontsize=12, color=GRAY)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.45, wspace=0.35,
                       top=0.93, bottom=0.04,
                       left=0.07, right=0.96)

# ── 4. Plot 1 – Correlation Heatmap ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])   # full-width top row
corr = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 10, 'weight': 'bold'},
            ax=ax1, cbar_kws={'shrink': 0.6})
ax1.set_title('Correlation Matrix – All Wellness Variables',
              fontsize=14, fontweight='bold', color=DARK, pad=12)
ax1.tick_params(axis='x', rotation=30, labelsize=9)
ax1.tick_params(axis='y', rotation=0,  labelsize=9)

# ── 5. Plot 2 – Sleep → Recovery Regression ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
x2, y2 = df['Sleep Hours'], df['Recovery Score']
slope2, intercept2, r2, p2, _ = stats.linregress(x2, y2)
x2_line = np.linspace(x2.min(), x2.max(), 100)

ax2.scatter(x2, y2, color=BRAND, alpha=0.55, s=55, edgecolors='white', linewidth=0.5)
ax2.plot(x2_line, slope2 * x2_line + intercept2, color=WARN, linewidth=2.5, label=f'y = {slope2:.2f}x + {intercept2:.1f}')
ax2.set_xlabel('Sleep Hours', fontsize=11, color=DARK)
ax2.set_ylabel('Recovery Score', fontsize=11, color=DARK)
ax2.set_title('Sleep Hours → Recovery Score', fontsize=13, fontweight='bold', color=DARK)
ax2.legend(fontsize=9)
r2_sq2 = r2**2
ax2.text(0.04, 0.92, f'r = {r2:.2f}   R² = {r2_sq2:.2f}   p = {p2:.3f}',
         transform=ax2.transAxes, fontsize=9, color=GRAY,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# ── 6. Plot 3 – Workout Intensity → Recovery Regression ──────────────────────
ax3 = fig.add_subplot(gs[1, 1])
x3, y3 = df['Workout Intensity'], df['Recovery Score']
slope3, intercept3, r3, p3, _ = stats.linregress(x3, y3)
x3_line = np.linspace(x3.min(), x3.max(), 100)

ax3.scatter(x3, y3, color=ACCENT, alpha=0.55, s=55, edgecolors='white', linewidth=0.5)
ax3.plot(x3_line, slope3 * x3_line + intercept3, color=WARN, linewidth=2.5, label=f'y = {slope3:.2f}x + {intercept3:.1f}')
ax3.set_xlabel('Workout Intensity (1=Low, 5=High)', fontsize=11, color=DARK)
ax3.set_ylabel('Recovery Score', fontsize=11, color=DARK)
ax3.set_title('Workout Intensity → Recovery Score', fontsize=13, fontweight='bold', color=DARK)
ax3.legend(fontsize=9)
r2_sq3 = r3**2
ax3.text(0.04, 0.92, f'r = {r3:.2f}   R² = {r2_sq3:.2f}   p = {p3:.3f}',
         transform=ax3.transAxes, fontsize=9, color=GRAY,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# ── 7. Plot 4 – Calories → Energy Level Regression ───────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
x4, y4 = df['Calories'], df['Energy Level']
slope4, intercept4, r4, p4, _ = stats.linregress(x4, y4)
x4_line = np.linspace(x4.min(), x4.max(), 100)

ax4.scatter(x4, y4, color='#7C3AED', alpha=0.55, s=55, edgecolors='white', linewidth=0.5)
ax4.plot(x4_line, slope4 * x4_line + intercept4, color=WARN, linewidth=2.5, label=f'y = {slope4:.4f}x + {intercept4:.1f}')
ax4.set_xlabel('Daily Calories', fontsize=11, color=DARK)
ax4.set_ylabel('Energy Level', fontsize=11, color=DARK)
ax4.set_title('Daily Calories → Energy Level', fontsize=13, fontweight='bold', color=DARK)
ax4.legend(fontsize=9)
r2_sq4 = r4**2
ax4.text(0.04, 0.92, f'r = {r4:.2f}   R² = {r2_sq4:.2f}   p = {p4:.3f}',
         transform=ax4.transAxes, fontsize=9, color=GRAY,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# ── 8. Plot 5 – Key Findings Summary ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

findings = [
    ('Sleep → Recovery',      f'r = {r2:.2f}',  p2,  'Positive: more sleep = better recovery'),
    ('Intensity → Recovery',  f'r = {r3:.2f}',  p3,  'Negative: harder workouts hurt recovery'),
    ('Calories → Energy',     f'r = {r4:.2f}',  p4,  'Weak: calorie count alone ≠ energy'),
]

ax5.text(0.05, 0.97, 'Key Findings', fontsize=14, fontweight='bold',
         color=DARK, transform=ax5.transAxes, va='top')
ax5.text(0.05, 0.88, 'Statistically significant relationships (p < 0.05) are marked ✓',
         fontsize=8, color=GRAY, transform=ax5.transAxes, va='top')

y_pos = 0.75
for title, r_val, p_val, insight in findings:
    sig = '✓ Significant' if p_val < 0.05 else '✗ Not significant'
    sig_color = ACCENT if p_val < 0.05 else '#EF4444'

    ax5.text(0.05, y_pos,      title,   fontsize=10, fontweight='bold', color=DARK,  transform=ax5.transAxes)
    ax5.text(0.05, y_pos-0.07, r_val,   fontsize=11, fontweight='bold', color=BRAND, transform=ax5.transAxes)
    ax5.text(0.30, y_pos-0.07, sig,     fontsize=9,  color=sig_color,               transform=ax5.transAxes)
    ax5.text(0.05, y_pos-0.14, insight, fontsize=8,  color=GRAY,                    transform=ax5.transAxes)
    line = plt.Line2D([0.03, 0.97], [y_pos-0.19, y_pos-0.19],
                      color='#E2E8F0', linewidth=1, transform=ax5.transAxes)
    ax5.add_line(line)
    y_pos -= 0.26

plt.savefig('/mnt/user-data/outputs/VitalQuest_Python_Analysis.png',
            dpi=150, bbox_inches='tight', facecolor='#F8FAFC')
print("Saved.")

# ── 9. Print summary stats ────────────────────────────────────────────────────
print("\n=== CORRELATION SUMMARY ===")
print(f"Sleep → Recovery:     r={r2:.3f}, R²={r2**2:.3f}, p={p2:.4f}")
print(f"Intensity → Recovery: r={r3:.3f}, R²={r3**2:.3f}, p={p3:.4f}")
print(f"Calories → Energy:    r={r4:.3f}, R²={r4**2:.3f}, p={p4:.4f}")
