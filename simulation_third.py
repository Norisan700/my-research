import json
import re
import math
import requests
import pandas as pd
from datetime import datetime
from scipy import stats
import numpy as np

# =====================================================================
# 1. 実験設定（Config）クラス
# =====================================================================
class GameConfig:
    N_SIMULATIONS = 3  # 学術水準として30〜50以上の独立試行を推奨
    CONDITIONS = [
        "A_Baseline",
        "B_Ablation_Sys1",
        "C_Ablation_Static",
        "D_Proposed"]
    MAX_DAYS = 5
    TURNS_PER_DAY = 6

    # 買い手設定
    BUYER_INIT_MONEY = 1000000
    BUYER_INIT_FOOD = 200
    BUYER_CONSUMPTION = 300
    BUYER_P_TAR = 500
    BUYER_P_RES = 2000
    BUYER_G_TAR = 400
    BUYER_G_RES = 250

    # 売り手設定
    SELLER_INIT_MONEY = 0
    SELLER_INIT_FOOD = 1500
    SELLER_PRODUCTION = 100
    SELLER_CONSUMPTION = 200
    SELLER_P_TAR = 1800
    SELLER_P_RES = 600
    SELLER_G_TAR = 200
    SELLER_G_RES = 400

    MODEL_NAME = "gemma4:31b"  # 2026年現在の実用的な高性能OSSモデルを指定

# =====================================================================
# 2. 効用関数 (Utility) とナッシュ交渉解 (NBS) の数理定義
# =====================================================================
def calc_buyer_utility(p, g):
    u_p = np.clip((GameConfig.BUYER_P_RES - p) / (GameConfig.BUYER_P_RES - GameConfig.BUYER_P_TAR), 0.0, 1.0)
    u_g = np.clip((g - GameConfig.BUYER_G_RES) / (GameConfig.BUYER_G_TAR - GameConfig.BUYER_G_RES), 0.0, 1.0)
    return 0.5 * u_p + 0.5 * u_g

def calc_seller_utility(p, g):
    u_p = np.clip((p - GameConfig.SELLER_P_RES) / (GameConfig.SELLER_P_TAR - GameConfig.SELLER_P_RES), 0.0, 1.0)
    u_g = np.clip((GameConfig.SELLER_G_RES - g) / (GameConfig.SELLER_G_RES - GameConfig.SELLER_G_TAR), 0.0, 1.0)
    return 0.5 * u_p + 0.5 * u_g

def get_nbs_solution():
    best_product = -1.0
    nbs_p, nbs_g = 0.0, 0.0
    for p in range(GameConfig.SELLER_P_RES, GameConfig.BUYER_P_RES + 1, 10):
        for g in range(GameConfig.BUYER_G_RES, GameConfig.SELLER_G_RES + 1, 10):
            u_b = calc_buyer_utility(p, g)
            u_s = calc_seller_utility(p, g)
            product = u_b * u_s
            if product > best_product:
                best_product = product
                nbs_p, nbs_g = p, g
    return nbs_p, nbs_g, calc_buyer_utility(nbs_p, nbs_g), calc_seller_utility(nbs_p, nbs_g)

# =====================================================================
# 3. モデレーターの重み（w1, w2）自動計算用の数理モデル
# =====================================================================
def calc_moderator_weights(food, role):
    """
    現在の食料在庫から生理的ストレスを擬似的に算出し、感情の重みをロジスティック曲線で決定する
    """
    f_req = GameConfig.BUYER_CONSUMPTION if role == "買い手" else GameConfig.SELLER_CONSUMPTION
    alpha = 100.0  # ストレス感受性パラメータ
    
    # 指数値のクリッピング（オーバーフロー防止）
    exponent = np.clip((food - f_req) / alpha, -20.0, 20.0)
    w1 = 1.0 / (1.0 + np.exp(exponent))  # 生存危機が迫るほど 1.0 に近づく
    w2 = 1.0 - w1
    return w1, w2

# =====================================================================
# 4. 数理計算エンジン（Pythonバックエンド）
# =====================================================================
def calculate_math_offer(agent, id, turn, max_turn, last_self_offer, last_opp_offer):
    t = turn
    T = max_turn
    
    if id == 1:    # 堅実な歩み寄り (Linear)
        P = agent.p_tar + (agent.p_res - agent.p_tar) * (t / T)
        G = agent.g_tar + (agent.g_res - agent.g_tar) * (t / T)
    elif id == 2:  # 戦略的ハッタリ (Boulware)
        if last_self_offer:
            P = last_self_offer["P"]
            G = last_self_offer["G"]
        else:
            P = agent.p_tar
            G = agent.g_tar
    elif id == 3:  # 協調的模倣 (Tit-for-Tat)
        if last_opp_offer and last_self_offer:
            P = (last_self_offer["P"] + last_opp_offer["P"]) / 2
            G = (last_self_offer["G"] + last_opp_offer["G"]) / 2
        else:
            P = agent.p_tar + (agent.p_res - agent.p_tar) * (t / T)
            G = agent.g_tar + (agent.g_res - agent.g_tar) * (t / T)
    elif id == 4:  # 不快感の表明 (Punish)
        if last_self_offer:
            direction = -1 if agent.role == "買い手" else 1  
            P = last_self_offer["P"] + (direction * abs(agent.p_tar - agent.p_res) * 0.1)
            G = last_self_offer["G"]
        else:
            P = agent.p_tar
            G = agent.g_tar
    elif id == 5 or id == 6:  # 焦燥の譲歩 / 受け入れ
        P = agent.p_res
        G = agent.g_res

    if agent.role == "買い手":
        P = min(P, agent.p_res)
        G = max(G, agent.g_res)
    else:
        P = max(P, agent.p_res)
        G = min(G, agent.g_res)

    P = round(P, 1)
    G = int(round(G))
    M = int(round(P * G))
    
    return {"P": P, "G": G, "M": M}

# =====================================================================
# 5. LLM（Ollama）への問い合わせと堅牢なパース・リトライ
# =====================================================================
def call_ollama(prompt, temp=0.3):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": GameConfig.MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temp}
    }
    try:
        response = requests.post(url, json=payload)
        return response.json().get("response", "")
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

def parse_robust_json(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        json_str = match.group(0) if match else text
        json_str = re.sub(r'//.*', '', json_str)  # コメント行の削除
        return json.loads(json_str)
    except:
        return {}

def get_system_scores(prompt, temp, n_items):
    """
    LLMから各選択肢の評価値（スコア）のリストを確実に取得するリトライループ
    """
    for attempt in range(3):
        current_temp = temp + attempt * 0.1
        output = call_ollama(prompt, temp=current_temp)
        parsed = parse_robust_json(output)
        
        if "scores" in parsed and isinstance(parsed["scores"], list) and len(parsed["scores"]) == n_items:
            try:
                # [-1.0, 1.0] に正規化・クリッピング
                scores = [max(-1.0, min(1.0, float(s))) for s in parsed["scores"]]
                return scores
            except ValueError:
                pass
    # 3回失敗した場合の完全フォールバック（全選択肢を均等に評価）
    return [0.0] * n_items

# =====================================================================
# 6. エージェントの思考プロセス（再設計版）
# =====================================================================
def agent_think(agent, day, turn, max_turn, history_text, last_opp_offer_text, predicted_offers, last_opp_offer_raw, condition):
    n_items = 6 if last_opp_offer_raw else 5
    tar_p, lim_p = agent.p_tar, agent.p_res
    
    strategy_names = {
        1: "堅実な歩み寄り（少しずつ歩み寄る）",
        2: "戦略的ハッタリ（前回の提案をキープ）",
        3: "協調的模倣（お互いの中間値を狙う）",
        4: "不快感の表明（相手へのお仕置きで条件悪化）",
        5: "焦燥の譲歩（自分の限界値を一気に提示）"
    }
    
    menu_text = ""
    for idx in [1, 2, 3, 4, 5]:
        off = predicted_offers[idx]
        menu_text += f"ID {idx}: {strategy_names[idx]} -> 【相手に『食料 {off['G']}g / 総額 {off['M']}円 (単価 {off['P']}円/g)』を提案】\n"
    if last_opp_offer_raw:
        menu_text += f"ID 6: 生存最優先の受け入れ -> 【相手の最新提案『食料 {last_opp_offer_raw['G']}g / 総額 {last_opp_offer_raw['M']}円』を丸呑みして今すぐ【合意】する】\n"

    # ==========================================
    # ① A_Baseline: 理性（System 2）のみの確定的選択
    # ==========================================
    if condition == "A_Baseline":
        prompt_sys2 = (
            f"あなたはサバイバル交渉中の{agent.role}の『理性と論理（システム2）』です。\n"
            f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, 交渉ターン {turn} / {max_turn}\n"
            f"自分のステータス: 所持金 {agent.money}円, 食料在庫 {agent.food}g\n"
            f"目標単価: {tar_p}円/g, 限界単価: {lim_p}円/g\n"
            f"本日のこれまでの交渉履歴:\n{history_text}\n\n"
            f"【指示】提示された{n_items}個の戦略選択肢（ID 1〜{n_items}）それぞれに対し、数理的有効性と論理的合理性の観点から、主観的な評価値を [-1.0（最悪）から 1.0（最高）] の範囲で算出して、リストとして出力してください。\n"
            f"【選択可能な行動】\n{menu_text}\n"
            f"出力は、以下のJSON形式のみとし、余計な説明やマークダウンは一切含めないでください：\n"
            f'{{"scores": [s1, s2, s3, s4, s5{", s6" if n_items == 6 else ""}]}}\n'
        )
        scores_sys2 = get_system_scores(prompt_sys2, temp=0.1, n_items=n_items)
        
        # 従来手法は「完全に確定的（Argmax）」として振る舞う（内的エントロピー = 0.0）
        selected_idx = int(np.argmax(scores_sys2))
        selected_id = selected_idx + 1
        
        print(f" > Baseline 理性選択: ID {selected_id} (スコア: {scores_sys2})")
        return selected_id, 0.0, 1.0, 0.0

    # ==========================================
    # ② B_Ablation_Sys1: 感情（System 1）のみの確率的選択
    # ==========================================
    elif condition == "B_Ablation_Sys1":
        prompt_sys1 = (
            f"あなたはサバイバル交渉中の{agent.role}の『生存本能と感情（システム1）』です。\n"
            # ...プロンプトの中身はそのまま
        )
        scores_sys1 = get_system_scores(prompt_sys1, temp=0.9, n_items=n_items)
        
        # QRE（w1=1.0, w2=0.0）
        V = np.array(scores_sys1)
        lam = 2.0
        exp_V = np.exp(np.clip(lam * V, -20.0, 20.0))
        probs = exp_V / np.sum(exp_V)
        selected_id = int(np.random.choice(range(1, n_items + 1), p=probs))
        
        print(f" > [Ablation_Sys1] 感情のみ選択: ID {selected_id}")
        return selected_id, 1.0, 0.0, 0.0

    # ==========================================
    # ③ C_Ablation_Static: 感情50%・理性50%の固定調停
    # ==========================================
    elif condition == "C_Ablation_Static":
        # 両システムからスコアを取得
        # (前述の prompt_sys1, prompt_sys2 を流用)
        scores_sys1 = get_system_scores(prompt_sys1, temp=0.9, n_items=n_items)
        scores_sys2 = get_system_scores(prompt_sys2, temp=0.1, n_items=n_items)
        
        # 重みを 0.5 固定にする（調停役の不在）
        w1, w2 = 0.5, 0.5
        V = w1 * np.array(scores_sys1) + w2 * np.array(scores_sys2)
        lam = 2.0
        exp_V = np.exp(np.clip(lam * V, -20.0, 20.0))
        probs = exp_V / np.sum(exp_V)
        selected_id = int(np.random.choice(range(1, n_items + 1), p=probs))
        
        weight_entropy = - (w1 * math.log(w1) + w2 * math.log(w2)) # 常に 0.6931
        print(f" > [Ablation_Static] 固定50:50選択: ID {selected_id}")
        return selected_id, w1, w2, weight_entropy

    # ==========================================
    # ④ D_Proposed: 提案手法（シグモイド調停あり）
    # ==========================================
    else:
        # 両システムからスコアを取得
        # (前述の prompt_sys1, prompt_sys2 を流用)
        scores_sys1 = get_system_scores(prompt_sys1, temp=0.9, n_items=n_items)
        scores_sys2 = get_system_scores(prompt_sys2, temp=0.1, n_items=n_items)
        
        # 生理的状況（餓死リスク）から動的に重みを決定
        w1, w2 = calc_moderator_weights(agent.food, agent.role)
        V = w1 * np.array(scores_sys1) + w2 * np.array(scores_sys2)
        lam = 2.0
        exp_V = np.exp(np.clip(lam * V, -20.0, 20.0))
        probs = exp_V / np.sum(exp_V)
        selected_id = int(np.random.choice(range(1, n_items + 1), p=probs))
        
        if w1 <= 0.0 or w2 <= 0.0:
            weight_entropy = 0.0
        else:
            weight_entropy = - (w1 * math.log(w1) + w2 * math.log(w2))
            
        print(f" > [Proposed] 動的調停選択: ID {selected_id} (w1={w1:.2f}, w2={w2:.2f})")
        return selected_id, w1, w2, weight_entropy

# =====================================================================
# 7. エージェントの状態管理クラス
# =====================================================================
class Agent:
    def __init__(self, role, money, food, p_tar, p_res, g_tar, g_res):
        self.role = role
        self.money = money
        self.food = food
        self.p_tar = p_tar  
        self.p_res = p_res  
        self.g_tar = g_tar  
        self.g_res = g_res  
        self.is_alive = True

# =====================================================================
# 8. シミュレーション管理メインループ
# =====================================================================
def run_experiment():
    all_logs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nbs_p, nbs_g, nbs_u_b, nbs_u_s = get_nbs_solution()

    for condition in GameConfig.CONDITIONS:
        for sim_id in range(1, GameConfig.N_SIMULATIONS + 1):
            print(f"\n🚀 [開始] 条件: {condition} | Sim: {sim_id}/{GameConfig.N_SIMULATIONS}")
            
            buyer = Agent("買い手", GameConfig.BUYER_INIT_MONEY, GameConfig.BUYER_INIT_FOOD, 
                          GameConfig.BUYER_P_TAR, GameConfig.BUYER_P_RES, GameConfig.BUYER_G_TAR, GameConfig.BUYER_G_RES)
            seller = Agent("売り手", GameConfig.SELLER_INIT_MONEY, GameConfig.SELLER_INIT_FOOD, 
                           GameConfig.SELLER_P_TAR, GameConfig.SELLER_P_RES, GameConfig.SELLER_G_TAR, GameConfig.SELLER_G_RES)

            for day in range(1, GameConfig.MAX_DAYS + 1):
                if not buyer.is_alive or not seller.is_alive: break
                
                seller.food += GameConfig.SELLER_PRODUCTION
                last_opp_offer_text = "まだ提案はありません"
                history_text = ""
                last_offers = {"買い手": None, "売り手": None}
                
                # 売り手の初期提示
                current_offer = {"P": seller.p_tar, "G": seller.g_tar, "M": int(seller.p_tar * seller.g_tar)}
                last_offers["売り手"] = current_offer
                last_opp_offer_text = f"食料 {current_offer['G']}g を 総額 {current_offer['M']}円 (単価 {current_offer['P']}円/g) で売る"
                history_text += f"[初期提示] 売り手: {last_opp_offer_text}\n"

                agreement_reached = False

                for turn in range(1, GameConfig.TURNS_PER_DAY + 1):
                    active_agent = buyer if turn % 2 == 1 else seller
                    passive_agent = seller if turn % 2 == 1 else buyer
                    
                    predicted_offers = {}
                    for idx in [1, 2, 3, 4, 5]:
                        predicted_offers[idx] = calculate_math_offer(
                            active_agent, idx, turn, GameConfig.TURNS_PER_DAY, 
                            last_offers[active_agent.role], last_offers[passive_agent.role]
                        )
                    
                    selected_id, w1, w2, entropy = agent_think(
                        active_agent, day, turn, GameConfig.TURNS_PER_DAY, 
                        history_text, last_opp_offer_text, predicted_offers, last_offers[passive_agent.role], condition
                    )
                    
                    if selected_id == 6:
                        final_offer = last_offers[passive_agent.role]
                        
                        # ガードレールチェック（破綻回避）
                        if active_agent.role == "買い手" and final_offer["M"] > buyer.money:
                            break
                        elif active_agent.role == "売り手" and final_offer["G"] > seller.food:
                            break
                        elif active_agent.role == "買い手" and final_offer["G"] > (seller.food - GameConfig.SELLER_CONSUMPTION):
                            break
                        
                        agreement_reached = True
                        buyer.money -= final_offer["M"]
                        buyer.food += final_offer["G"]
                        seller.money += final_offer["M"]
                        seller.food -= final_offer["G"]

                        # NBS（ナッシュ交渉解）乖離度の計算
                        u_b_act = calc_buyer_utility(final_offer["P"], final_offer["G"])
                        u_s_act = calc_seller_utility(final_offer["P"], final_offer["G"])
                        nbs_dist = math.sqrt((u_b_act - nbs_u_b)**2 + (u_s_act - nbs_u_s)**2)

                        all_logs.append({
                            "sim_id": sim_id, "condition": condition, "day": day, "turn": turn,
                            "role": active_agent.role, "action": "accept", "food": final_offer["G"], "price": final_offer["M"],
                            "unit_price": final_offer["P"], "conflict_score": entropy, "nbs_distance": nbs_dist,
                            "buyer_money": buyer.money, "buyer_food": buyer.food, "seller_money": seller.money, "seller_food": seller.food
                        })
                        break
                        
                    else:
                        calculated_offer = predicted_offers[selected_id]
                        last_offers[active_agent.role] = calculated_offer
                        last_opp_offer_text = f"食料 {calculated_offer['G']}g を 総額 {calculated_offer['M']}円 (単価 {calculated_offer['P']}円/g) で取引したい"
                        history_text += f"[ターン{turn}] {active_agent.role}: {last_opp_offer_text}\n"

                        all_logs.append({
                            "sim_id": sim_id, "condition": condition, "day": day, "turn": turn,
                            "role": active_agent.role, "action": "propose", "food": calculated_offer["G"], "price": calculated_offer["M"],
                            "unit_price": calculated_offer["P"], "conflict_score": entropy, "nbs_distance": None,
                            "buyer_money": buyer.money, "buyer_food": buyer.food, "seller_money": seller.money, "seller_food": seller.food
                        })

                # 夜の消費処理と生存判定
                buyer.food -= GameConfig.BUYER_CONSUMPTION
                seller.food -= GameConfig.SELLER_CONSUMPTION
                if buyer.food < 0: buyer.is_alive = False
                if seller.food < 0: seller.is_alive = False

    df = pd.DataFrame(all_logs)
    filename = f"experiment_v3_results_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    return filename

# =====================================================================
# 9. 自動データ集計・U検定モジュール
# =====================================================================
def analyze_data_v3(csv_filename):
    df = pd.read_csv(csv_filename)
    print("\n" + "="*50 + "\n📊 統計解析レポート (IEEEレベル対応)\n" + "="*50)

    # ① 生存日数のU検定
    surv_df = df.groupby(['condition', 'sim_id'])['day'].max().reset_index()
    a_surv = surv_df[surv_df['condition'] == 'A_Baseline']['day']
    b_surv = surv_df[surv_df['condition'] == 'B_Proposed']['day']
    _, p_surv = stats.mannwhitneyu(a_surv, b_surv, alternative='two-sided')
    print(f"生存日数 (Days Survived):")
    print(f"  Baseline: {a_surv.mean():.2f}日 | Proposed: {b_surv.mean():.2f}日 (p = {p_surv:.5f})")

    # ② 合意率のU検定
    total_days = df.groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
    accept_days = df[df['action'] == 'accept'].groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
    ag_df = pd.merge(total_days, accept_days, on=['condition', 'sim_id'], how='left').fillna(0)
    ag_df['rate'] = ag_df['day_y'] / ag_df['day_x']
    a_rate = ag_df[ag_df['condition'] == 'A_Baseline']['rate']
    b_rate = ag_df[ag_df['condition'] == 'B_Proposed']['rate']
    _, p_rate = stats.mannwhitneyu(a_rate, b_rate, alternative='two-sided')
    print(f"合意率 (Agreement Rate):")
    print(f"  Baseline: {a_rate.mean()*100:.1f}% | Proposed: {b_rate.mean()*100:.1f}% (p = {p_rate:.5f})")

    # ③ 葛藤度エントロピーの集計
    b_entropy = df[(df['condition'] == 'B_Proposed') & (df['action'] == 'propose')]['conflict_score']
    print(f"提案手法の平均内的葛藤度 (Shannon Entropy): {b_entropy.mean():.4f} (理論的最大値: 0.6931)")

    # ④ NBS（ナッシュ交渉解）距離のU検定
    a_nbs = df[(df['condition'] == 'A_Baseline') & (df['action'] == 'accept')]['nbs_distance'].dropna()
    b_nbs = df[(df['condition'] == 'B_Proposed') & (df['action'] == 'accept')]['nbs_distance'].dropna()
    _, p_nbs = stats.mannwhitneyu(a_nbs, b_nbs, alternative='two-sided')
    print(f"NBSからの乖離度 (Distance to NBS - 小さいほど良い):")
    print(f"  Baseline: {a_nbs.mean():.4f} | Proposed: {b_nbs.mean():.4f} (p = {p_nbs:.5f})")
    print("="*50)

if __name__ == "__main__":
    filename = run_experiment()
    analyze_data_v3(filename)