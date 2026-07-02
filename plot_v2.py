import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =====================================================================
# ★ ご自身のフォルダ内にある最新のCSVファイル名に書き換えてください
# =====================================================================
csv_filename = "experiment_v2_results_20260626_202635.csv" 

# 日本語フォントの設定（文字化け対策）
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid", context="talk")

# データの読み込み
df = pd.read_csv(csv_filename)
colors = ["#A9A9A9", "#4C72B0"]
labels = ['Baseline\n(Logical CoT)', 'Proposed\n(Dual-Process)']

# ---------------------------------------------------
# ① 生存日数の比較グラフ (Bar Plot)
# ---------------------------------------------------
plt.figure(figsize=(6, 5))
survival_df = df.groupby(['condition', 'sim_id'])['day'].max().reset_index()
sns.barplot(x='condition', y='day', data=survival_df, capsize=.1, errorbar='sd', palette=colors)
plt.title("Average Survival Days", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Agent Architecture", fontsize=12)
plt.ylabel("Days Survived (Max 5)", fontsize=12)
plt.xticks([0, 1], labels)
plt.ylim(0, 5)
plt.tight_layout()
plt.savefig("graph_survival_days.png", dpi=300)
plt.close()

# ---------------------------------------------------
# ② 最終合意率の比較グラフ (Bar Plot)
# ---------------------------------------------------
plt.figure(figsize=(6, 5))
total_days = df.groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
accept_days = df[df['action'] == 'accept'].groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
ag_rate_df = pd.merge(total_days, accept_days, on=['condition', 'sim_id'], how='left').fillna(0)
ag_rate_df['agreement_rate'] = ag_rate_df['day_y'] / ag_rate_df['day_x'] * 100

sns.barplot(x='condition', y='agreement_rate', data=ag_rate_df, capsize=.1, errorbar='sd', palette=colors)
plt.title("Agreement Rate per Day (%)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Agent Architecture", fontsize=12)
plt.ylabel("Agreement Rate (%)", fontsize=12)
plt.xticks([0, 1], labels)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("graph_agreement_rate.png", dpi=300)
plt.close()

# ---------------------------------------------------
# ③ NBSからの乖離度の比較グラフ (Box Plot)
# ---------------------------------------------------
plt.figure(figsize=(6, 5))
accept_df = df[df['action'] == 'accept'].dropna(subset=['nbs_distance'])
sns.boxplot(x='condition', y='nbs_distance', data=accept_df, palette=colors, showfliers=False)
plt.title("Distance to NBS (Lower is Better)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Agent Architecture", fontsize=12)
plt.ylabel("Distance to Nash Solution", fontsize=12)
plt.xticks([0, 1], labels)
plt.tight_layout()
plt.savefig("graph_nbs_distance.png", dpi=300)
plt.close()

# ---------------------------------------------------
# ④ 【新規】日数経過に伴う内的葛藤度（エントロピー）の推移
# ---------------------------------------------------
plt.figure(figsize=(8, 5))
# 提案手法の提案ターンのみを抽出
prop_entropy = df[(df['condition'] == 'B_Proposed') & (df['action'] == 'propose')]
entropy_trend = prop_entropy.groupby('day')['conflict_score'].mean().reset_index()

sns.lineplot(x='day', y='conflict_score', data=entropy_trend, marker='o', color='#4C72B0', linewidth=2.5)
plt.title("Trend of Internal Conflict (Proposed Method)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Day (Survival Progress)", fontsize=12)
plt.ylabel("Average Conflict Entropy", fontsize=12)
plt.ylim(0, 0.7) # エントロピーの最大値は0.693
plt.xticks(range(1, 6))
plt.tight_layout()
plt.savefig("graph_entropy_trend.png", dpi=300)
plt.close()

print("📁 グラフ画像の生成が完了しました！")