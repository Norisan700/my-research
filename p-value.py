import pandas as pd
import numpy as np
from scipy import stats

# CSVファイルの読み込み
csv_filename = "experiment_results_20260601_175931.csv" # 実際のファイル名に変更
df = pd.read_csv(csv_filename)

print("🔬 マン・ホイットニーのU検定 結果レポート\n" + "="*45)

# ---------------------------------------------------
# 1. 生存日数 (Survival Days) の検定
# ---------------------------------------------------
survival_df = df.groupby(['condition', 'sim_id'])['day'].max().reset_index()
a_survival = survival_df[survival_df['condition'] == 'A_Baseline']['day']
b_survival = survival_df[survival_df['condition'] == 'B_Proposed']['day']

# U検定の実行 (alternative='two-sided' で両側検定)
u_stat, p_val_surv = stats.mannwhitneyu(a_survival, b_survival, alternative='two-sided')
print(f"[1] 生存日数")
print(f"  A平均: {a_survival.mean():.2f}日 vs B平均: {b_survival.mean():.2f}日")
print(f"  p値 = {p_val_surv:.4f}")

# ---------------------------------------------------
# 2. 最終合意率 (Agreement Rate) の検定
# ---------------------------------------------------
total_days = df.groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
accept_days = df[df['action'] == 'accept'].groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
ag_rate_df = pd.merge(total_days, accept_days, on=['condition', 'sim_id'], how='left').fillna(0)
ag_rate_df['agreement_rate'] = ag_rate_df['day_y'] / ag_rate_df['day_x']

a_agree = ag_rate_df[ag_rate_df['condition'] == 'A_Baseline']['agreement_rate']
b_agree = ag_rate_df[ag_rate_df['condition'] == 'B_Proposed']['agreement_rate']

u_stat, p_val_agree = stats.mannwhitneyu(a_agree, b_agree, alternative='two-sided')
print(f"\n[2] 最終合意率")
print(f"  A平均: {a_agree.mean():.3f} vs B平均: {b_agree.mean():.3f}")
print(f"  p値 = {p_val_agree:.4f}")

# ---------------------------------------------------
# 3. 葛藤度スコア (Conflict Score) の検定
# ---------------------------------------------------
propose_df = df[df['action'] == 'propose']
conflict_mean_df = propose_df.groupby(['condition', 'sim_id'])['conflict_score'].mean().reset_index()

a_conflict = conflict_mean_df[conflict_mean_df['condition'] == 'A_Baseline']['conflict_score']
b_conflict = conflict_mean_df[conflict_mean_df['condition'] == 'B_Proposed']['conflict_score']

u_stat, p_val_conf = stats.mannwhitneyu(a_conflict, b_conflict, alternative='two-sided')
print(f"\n[3] 葛藤度スコア (CS_t)")
print(f"  A平均: {a_conflict.mean():.4f} vs B平均: {b_conflict.mean():.4f}")
print(f"  p値 = {p_val_conf:.4f}")

print("="*45)

# --- 判定と論文表記のヒント ---
def print_significance(p):
    if p < 0.01:
        return "強い有意差あり (**) -> 論文で「極めて確実な差」と主張可能"
    elif p < 0.05:
        return "有意差あり (*) -> 論文で「明確な差がある」と主張可能"
    elif p < 0.1:
        return "有意傾向あり (+) -> 「差がある傾向が見られた」と記載"
    else:
        return "有意差なし (n.s.) -> 「偶然の誤差の範囲内」"

print("\n💡 論文執筆のための判定：")
print(f"・生存日数: {print_significance(p_val_surv)}")
print(f"・合意率　: {print_significance(p_val_agree)}")
print(f"・葛藤度　: {print_significance(p_val_conf)}")