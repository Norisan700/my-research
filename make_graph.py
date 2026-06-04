import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 論文・スライド用 グラフ生成スクリプト
# ==========================================
# CSVファイルの読み込み（※実際のファイル名に合わせて変更してください）
csv_filename = "experiment_results_20260601_175931.csv"
df = pd.read_csv(csv_filename)

# 学術論文向けの美しいスタイル設定
sns.set_theme(style="whitegrid", context="talk")
colors = ["#A9A9A9", "#4C72B0"] # ベースラインはグレー、提案手法は強調のブルー
labels = ['Baseline\n(Standard LLM)', 'Proposed\n(Dual-Process)']

# ---------------------------------------------------
# ① マクロ指標：生存日数 (Survival Days) の棒グラフ
# ---------------------------------------------------
plt.figure(figsize=(8, 6))
survival_df = df.groupby(['condition', 'sim_id'])['day'].max().reset_index()
sns.barplot(x='condition', y='day', data=survival_df, capsize=.1, errorbar='sd', palette=colors)

plt.title("Average Survival Days", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Agent Architecture", fontsize=14)
plt.ylabel("Days Survived (Max 5)", fontsize=14)
plt.xticks([0, 1], labels)
plt.ylim(0, 6)
plt.tight_layout()
plt.savefig("graph_survival_days.png", dpi=300) # 高解像度で保存
plt.show()

# ---------------------------------------------------
# ② マクロ指標：最終合意率 (Agreement Rate) の棒グラフ
# ---------------------------------------------------
plt.figure(figsize=(8, 6))
total_days = df.groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
accept_days = df[df['action'] == 'accept'].groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
ag_rate_df = pd.merge(total_days, accept_days, on=['condition', 'sim_id'], how='left').fillna(0)
ag_rate_df['agreement_rate'] = ag_rate_df['day_y'] / ag_rate_df['day_x'] * 100 # %に変換

sns.barplot(x='condition', y='agreement_rate', data=ag_rate_df, capsize=.1, errorbar='sd', palette=colors)
plt.title("Agreement Rate per Day (%)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Agent Architecture", fontsize=14)
plt.ylabel("Agreement Rate (%)", fontsize=14)
plt.xticks([0, 1], labels)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("graph_agreement_rate.png", dpi=300)
plt.show()

# ---------------------------------------------------
# ③ 内部指標：葛藤度スコア (Conflict Score) の箱ひげ図
# ---------------------------------------------------
plt.figure(figsize=(8, 6))
propose_df = df[df['action'] == 'propose']

sns.boxplot(x='condition', y='conflict_score', data=propose_df, palette=colors, showfliers=False)
plt.title("Volatility of Price Proposals (Conflict Score)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Agent Architecture", fontsize=14)
plt.ylabel("Conflict Score (Log Return of Unit Price)", fontsize=14)
plt.xticks([0, 1], labels)
plt.tight_layout()
plt.savefig("graph_conflict_score.png", dpi=300)
plt.show()