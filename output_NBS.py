import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# ナッシュ交渉解 (NBS) 乖離度 算出スクリプト
# ==========================================
csv_filename = "experiment_results_20260601_175931.csv" # 実際のファイル名に変更
df = pd.read_csv(csv_filename)

# --- 1. 効用(Utility)計算関数 ---
def calc_buyer_utility(p, g):
    # 単価効用 (500円で1.0, 2000円で0.0)
    u_p = np.clip((2000 - p) / (2000 - 500), 0.0, 1.0)
    # 数量効用 (400gで1.0, 250gで0.0)
    u_g = np.clip((g - 250) / (400 - 250), 0.0, 1.0)
    return 0.5 * u_p + 0.5 * u_g

def calc_seller_utility(p, g):
    # 単価効用 (1800円で1.0, 600円で0.0)
    u_p = np.clip((p - 600) / (1800 - 600), 0.0, 1.0)
    # 数量効用 (200gで1.0, 400gで0.0)
    u_g = np.clip((400 - g) / (400 - 200), 0.0, 1.0)
    return 0.5 * u_p + 0.5 * u_g

# --- 2. 理論上のナッシュ交渉解(NBS)の探索 ---
# 妥当なP(600〜2000)とG(150〜500)の全組み合わせからNBSを探す
best_product = -1.0
nbs_u_b, nbs_u_s = 0.0, 0.0

for p in range(600, 2001, 10):
    for g in range(150, 501, 10):
        u_b = calc_buyer_utility(p, g)
        u_s = calc_seller_utility(p, g)
        product = u_b * u_s
        if product > best_product:
            best_product = product
            nbs_u_b, nbs_u_s = u_b, u_s

print(f"🎯 理論上のナッシュ交渉解(NBS): Buyer Utility={nbs_u_b:.2f}, Seller Utility={nbs_u_s:.2f}")

# --- 3. 実際の合意データとの乖離度計算 ---
# 合意（accept）に至った行のみを抽出
accept_df = df[df['action'] == 'accept'].copy()

# 単価(unit_price)を計算（欠損値対策）
accept_df['unit_price'] = accept_df['price_offer'] / accept_df['food_offer']

# 実際の効用を計算
accept_df['buyer_utility'] = accept_df.apply(lambda row: calc_buyer_utility(row['unit_price'], row['food_offer']), axis=1)
accept_df['seller_utility'] = accept_df.apply(lambda row: calc_seller_utility(row['unit_price'], row['food_offer']), axis=1)

# ユークリッド距離（NBSからの乖離度）の計算
accept_df['distance_to_nbs'] = np.sqrt(
    (accept_df['buyer_utility'] - nbs_u_b)**2 + 
    (accept_df['seller_utility'] - nbs_u_s)**2
)

# --- 4. 解析結果の表示とグラフ化 ---
stats = accept_df.groupby('condition')['distance_to_nbs'].mean().reset_index()
print("\n📊 NBSからの乖離度（小さいほど数学的最適解に近い）：")
print(stats.to_string(index=False))

# 箱ひげ図の作成
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(8, 6))
colors = ["#A9A9A9", "#4C72B0"]
sns.boxplot(x='condition', y='distance_to_nbs', data=accept_df, palette=colors, showfliers=False)

plt.title("Distance to Nash Bargaining Solution (NBS)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Agent Architecture", fontsize=14)
plt.ylabel("Distance (Lower is Better)", fontsize=14)
plt.xticks([0, 1], ['Baseline\n(Standard LLM)', 'Proposed\n(Dual-Process)'])
plt.tight_layout()
plt.savefig("graph_nbs_distance.png", dpi=300)
plt.show()