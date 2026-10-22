import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# スタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# 生成された要約CSVを読み込み
summary_file = "tournament_summary_fair_20261020_215325.csv"
df = pd.read_csv(summary_file)

# 16マスのピボットテーブル作成
order = ["A_Baseline", "B_Ablation_Sys1", "C_Ablation_Static", "D_Proposed"]
matrix = df.pivot_table(
    index='buyer_cond', 
    columns='seller_cond', 
    values='social_survival', 
    aggfunc='mean'
).reindex(index=order, columns=order)

# ==========================================
# 1. 16マスの社会的生存日数ヒートマップ
# ==========================================
plt.figure(figsize=(7, 6))
sns.heatmap(
    matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    cbar_kws={'label': 'Social Survival Days (Max 10)'},
    linewidths=1,
    linecolor='white'
)
plt.title("Tournament Payoff Matrix: Social Survival Days", fontsize=13, fontweight='bold')
plt.xlabel("Seller Architecture", fontsize=11, fontweight='bold')
plt.ylabel("Buyer Architecture", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("graph_tournament_heatmap.png", dpi=300)
plt.close()

# ==========================================
# 2. 各モデルの総合適応力（棒グラフ）
# ==========================================
buyer_scores = df.groupby('buyer_cond')['social_survival'].mean()
seller_scores = df.groupby('seller_cond')['social_survival'].mean()
total_scores = ((buyer_scores + seller_scores) / 2).reindex(order).reset_index()
total_scores.columns = ['Model', 'Average_Social_Survival']

plt.figure(figsize=(6, 4.5))
ax = sns.barplot(x='Model', y='Average_Social_Survival', data=total_scores, palette='coolwarm')
plt.title("Overall Social Adaptability (Tournament)", fontsize=12, fontweight='bold')
plt.ylabel("Average Social Survival Days (/10)")
plt.xlabel("Agent Model")
plt.ylim(0, 6)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}d", (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig("graph_tournament_totals.png", dpi=300)
plt.close()

print("🎉 ヒートマップと総合適応力グラフの生成が完了しました！")