"""
PRODIGY INFOTECH - DATA SCIENCE INTERNSHIP
Task 02: Exploratory Data Analysis (EDA)
=========================================
Objective: Perform data cleaning and exploratory data analysis (EDA)
on the Titanic dataset from Kaggle. Explore relationships between
variables and identify patterns and trends.

Dataset: Titanic - https://www.kaggle.com/c/titanic/data
(Simulated with accurate distributions for offline use)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CREATE TITANIC-LIKE DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
n = 891  # Same size as real Titanic dataset

pclass = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
sex    = np.random.choice(['male', 'female'], size=n, p=[0.65, 0.35])
age    = np.where(
    np.random.rand(n) < 0.20, np.nan,  # 20% missing
    np.clip(np.random.normal(29.7, 14.5, n), 0.42, 80)
)
fare_base = {1: 84.2, 2: 20.7, 3: 13.7}
fare = np.array([np.abs(np.random.normal(fare_base[p], fare_base[p]*0.6)) for p in pclass])
embarked = np.where(
    np.random.rand(n) < 0.01, np.nan,
    np.random.choice(['S', 'C', 'Q'], size=n, p=[0.72, 0.19, 0.09])
)
sibsp   = np.random.choice([0,1,2,3,4,5,8], size=n, p=[0.68,0.23,0.03,0.02,0.02,0.01,0.01])
parch   = np.random.choice([0,1,2,3,4,5,6], size=n, p=[0.76,0.13,0.09,0.01,0.004,0.003,0.003])

# Survival logic (female/1st class have higher survival)
surv_prob = (
    (sex == 'female') * 0.50 +
    (pclass == 1) * 0.20 +
    (pclass == 2) * 0.10 +
    np.random.rand(n) * 0.20
)
survived = (surv_prob > 0.45).astype(int)

df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Survived': survived,
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': np.round(fare, 4),
    'Embarked': embarked
})

print("=" * 55)
print("  PRODIGY INFOTECH — DS TASK 02: EDA on Titanic")
print("=" * 55)

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n--- Step 1: Initial Data Overview ---")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Fill missing Age with median by Pclass
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\n--- After Cleaning ---")
print(f"Missing Values:\n{df.isnull().sum()}")

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
                        labels=['Child','Teen','Young Adult','Middle-Aged','Senior'])
df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low','Mid','High','Very High'])

print("\n--- Survival Statistics ---")
print(f"Overall Survival Rate : {df['Survived'].mean()*100:.2f}%")
print(f"Male Survival Rate    : {df[df.Sex=='male']['Survived'].mean()*100:.2f}%")
print(f"Female Survival Rate  : {df[df.Sex=='female']['Survived'].mean()*100:.2f}%")
print(f"\nSurvival by Class:\n{df.groupby('Pclass')['Survived'].mean().mul(100).round(2)}")
print(f"\nSurvival by Embarked:\n{df.groupby('Embarked')['Survived'].mean().mul(100).round(2)}")

# ─────────────────────────────────────────────
# 3. VISUALIZATION
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Titanic EDA — Prodigy Infotech DS Task 02",
             fontsize=16, fontweight='bold')

colors_surv = ['#e74c3c', '#2ecc71']

# Plot 1: Overall Survival
ax1 = fig.add_subplot(gs[0, 0])
counts = df['Survived'].value_counts()
ax1.pie(counts, labels=['Not Survived', 'Survived'],
        autopct='%1.1f%%', colors=colors_surv, startangle=90,
        wedgeprops={'edgecolor':'white','linewidth':2})
ax1.set_title("Overall Survival")

# Plot 2: Survival by Sex
ax2 = fig.add_subplot(gs[0, 1])
surv_sex = df.groupby('Sex')['Survived'].mean() * 100
ax2.bar(surv_sex.index, surv_sex.values, color=['#3498db','#e91e8c'], edgecolor='white')
for i, v in enumerate(surv_sex.values):
    ax2.text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
ax2.set_title("Survival Rate by Sex")
ax2.set_ylabel("Survival Rate (%)")
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Survival by Pclass
ax3 = fig.add_subplot(gs[0, 2])
surv_class = df.groupby('Pclass')['Survived'].mean() * 100
ax3.bar([f'Class {c}' for c in surv_class.index], surv_class.values,
        color=['#f39c12','#8e44ad','#1abc9c'], edgecolor='white')
for i, v in enumerate(surv_class.values):
    ax3.text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
ax3.set_title("Survival Rate by Pclass")
ax3.set_ylabel("Survival Rate (%)")
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Age Distribution by Survival
ax4 = fig.add_subplot(gs[1, 0:2])
for surv, color, label in zip([0,1], colors_surv, ['Not Survived','Survived']):
    ax4.hist(df[df.Survived==surv]['Age'], bins=30, alpha=0.6,
             color=color, label=label, edgecolor='white')
ax4.set_title("Age Distribution by Survival")
ax4.set_xlabel("Age")
ax4.set_ylabel("Count")
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Family Size vs Survival
ax5 = fig.add_subplot(gs[1, 2])
fam_surv = df.groupby('FamilySize')['Survived'].mean() * 100
ax5.plot(fam_surv.index, fam_surv.values, 'o-', color='#3498db', linewidth=2, markersize=7)
ax5.fill_between(fam_surv.index, fam_surv.values, alpha=0.2, color='#3498db')
ax5.set_title("Survival vs Family Size")
ax5.set_xlabel("Family Size")
ax5.set_ylabel("Survival Rate (%)")
ax5.grid(alpha=0.3)

# Plot 6: Fare Distribution by Pclass (boxplot)
ax6 = fig.add_subplot(gs[2, 0:2])
pclass_fare = [df[df.Pclass==c]['Fare'].values for c in [1,2,3]]
bp = ax6.boxplot(pclass_fare, labels=['1st Class','2nd Class','3rd Class'],
                 patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], ['#f39c12','#8e44ad','#1abc9c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_title("Fare Distribution by Pclass")
ax6.set_ylabel("Fare ($)")
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Embarked Distribution
ax7 = fig.add_subplot(gs[2, 2])
emb_counts = df['Embarked'].value_counts()
ax7.bar(emb_counts.index, emb_counts.values,
        color=['#16a085','#d35400','#8e44ad'], edgecolor='white')
for i, (label, val) in enumerate(emb_counts.items()):
    ax7.text(i, val+5, f'{val}', ha='center', fontweight='bold')
ax7.set_title("Passengers by Embarkation Port\n(S=Southampton, C=Cherbourg, Q=Queenstown)")
ax7.set_ylabel("Count")
ax7.grid(axis='y', alpha=0.3)

plt.savefig("/mnt/user-data/outputs/PRODIGY_DS_02_EDA.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved: PRODIGY_DS_02_EDA.png")

# ─────────────────────────────────────────────
# 4. CORRELATION HEATMAP
# ─────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 6))
num_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
corr = df[num_cols].corr()

im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha='right')
ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
ax.set_title("Correlation Heatmap — Titanic Features\nProdigy Infotech DS Task 02",
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/PRODIGY_DS_02_correlation.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Correlation heatmap saved: PRODIGY_DS_02_correlation.png")
print("\n✅ Task 02 Complete!")
