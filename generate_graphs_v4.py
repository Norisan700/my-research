import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# スタイルを学術論文向けに設定
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# CSVの読み込み（ファイル名はご自身のものに変更してください）
csv_filename = "experiment_v3_results_20260702_131146.csv"
df = pd.read_csv(csv_filename)

# 4条件の順序を固定
order = ["A_Baseline", "B_Ablation_Sys1", "C_Ablation_Static", "D_Proposed"]

# =====================================================================
# 1. 平均生存日数の比較（棒グラフ）
# =====================================================================
plt.figure(figsize=(6, 4.5))
surv_df = df.groupby(['condition', 'sim_id'])['day'].max().reset_index()

ax1 = sns.barplot(
    x='condition', y='day', data=surv_df, 
    order=order, capsize=0.1, errorbar='ci', 
    hue='condition', legend=False, palette='muted'
)
plt.title("Comparison of Average Survival Days (N=100)", fontsize=12, fontweight='bold')
plt.ylabel("Days Survived (Max 5)")
plt.xlabel("Agent Conditions")
plt.ylim(0, 5)

# バーの上に数値を表示
for p in ax1.patches:
    height = p.get_height()
    if height > 0:
        ax1.annotate(f"{height:.2f}d", (p.get_x() + p.get_width() / 2., height + 0.1),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig("graph_ablation_survival.png", dpi=300)
plt.close()


# =====================================================================
# 2. 最終合意率の比較（棒グラフ）
# =====================================================================
plt.figure(figsize=(6, 4.5))
total_days = df.groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
accept_days = df[df['action'] == 'accept'].groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
ag_df = pd.merge(total_days, accept_days, on=['condition', 'sim_id'], how='left').fillna(0)
ag_df['rate'] = (ag_df['day_y'] / ag_df['day_x']) * 100

ax2 = sns.barplot(
    x='condition', y='rate', data=ag_df, 
    order=order, capsize=0.1, errorbar='ci', 
    hue='condition', legend=False, palette='muted'
)
plt.title("Agreement Rate per Day (%)", fontsize=12, fontweight='bold')
plt.ylabel("Agreement Rate (%)")
plt.xlabel("Agent Conditions")
plt.ylim(0, 100)

# バーの上にパーセンテージを表示
for p in ax2.patches:
    height = p.get_height()
    if height > 0:
        ax2.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 1),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig("graph_ablation_agreement.png", dpi=300)
plt.close()


# =====================================================================
# 3. ナッシュ交渉解からの乖離度（箱ひげ図）
# =====================================================================
plt.figure(figsize=(6, 4.5))
nbs_df = df[df['action'] == 'accept'].dropna(subset=['nbs_distance'])

sns.boxplot(
    x='condition', y='nbs_distance', data=nbs_df, 
    order=order, hue='condition', legend=False, palette='muted', width=0.5
)
plt.title("Distance to Nash Bargaining Solution (NBS)", fontsize=12, fontweight='bold')
plt.ylabel("Distance (Lower is Better)")
plt.xlabel("Agent Conditions")

plt.tight_layout()
plt.savefig("graph_ablation_nbs_distance.png", dpi=300)
plt.close()


# =====================================================================
# 4. 内的葛藤度（エントロピー）の推移の比較（折れ線グラフ）
# =====================================================================
plt.figure(figsize=(7, 4.5))
entropy_df = df[df['action'] == 'propose']
entropy_sub = entropy_df[entropy_df['condition'].isin(["C_Ablation_Static", "D_Proposed"])]

sns.lineplot(
    x='day', y='conflict_score', hue='condition', data=entropy_sub,
    hue_order=["C_Ablation_Static", "D_Proposed"], marker='o', errorbar='ci', palette='Set1'
)
plt.title("Dynamic vs. Static Conflict Entropy Over Days", fontsize=12, fontweight='bold')
plt.ylabel("Internal Conflict Entropy (Shannon)")
plt.xlabel("Day (Survival Progress)")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 0.8)
plt.legend(title="Condition")

plt.tight_layout()
plt.savefig("graph_entropy_comparison.png", dpi=300)
plt.close()

print("🎉 学術論文用の全4枚のグラフ生成が完了しました！")