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
    N_SIMULATIONS = 1  # ★本番実験時は 30〜50 に設定してください
    CONDITIONS = ["A_Baseline", "B_Proposed"]
    MAX_DAYS = 5
    TURNS_PER_DAY = 6

    # 買い手設定
    BUYER_INIT_MONEY = 1000000
    BUYER_INIT_FOOD = 200
    BUYER_CONSUMPTION = 300
    BUYER_TARGET_P = 500
    BUYER_LIMIT_P = 2000
    BUYER_TARGET_G = 400
    BUYER_LIMIT_G = 250

    # 売り手設定
    SELLER_INIT_MONEY = 0
    SELLER_INIT_FOOD = 1500
    SELLER_PRODUCTION = 100
    SELLER_CONSUMPTION = 200
    SELLER_TARGET_P = 1800
    SELLER_LIMIT_P = 600
    SELLER_TARGET_G = 200
    SELLER_LIMIT_G = 400

    MODEL_NAME = "gemma4:31b" # ご指定のモデル名

# =====================================================================
# 2. 効用関数 (Utility) とナッシュ交渉解 (NBS) の数理定義
# =====================================================================
def calc_buyer_utility(p, g):
    # 単価効用 (500円で1.0, 2000円で0.0)
    u_p = np.clip((GameConfig.BUYER_LIMIT_P - p) / (GameConfig.BUYER_LIMIT_P - GameConfig.BUYER_TARGET_P), 0.0, 1.0)
    # 数量効用 (400gで1.0, 250gで0.0)
    u_g = np.clip((g - GameConfig.BUYER_LIMIT_G) / (GameConfig.BUYER_TARGET_G - GameConfig.BUYER_LIMIT_G), 0.0, 1.0)
    return 0.5 * u_p + 0.5 * u_g

def calc_seller_utility(p, g):
    # 単価効用 (1800円で1.0, 600円で0.0)
    u_p = np.clip((p - GameConfig.SELLER_LIMIT_P) / (GameConfig.SELLER_TARGET_P - GameConfig.SELLER_LIMIT_P), 0.0, 1.0)
    # 数量効用 (200gで1.0, 400gで0.0)
    u_g = np.clip((GameConfig.SELLER_LIMIT_G - g) / (GameConfig.SELLER_LIMIT_G - GameConfig.SELLER_TARGET_G), 0.0, 1.0)
    return 0.5 * u_p + 0.5 * u_g

# 理論上のナッシュ交渉解 (NBS) を探索する
def find_nash_solution():
    best_product = -1.0
    nbs_p, nbs_g = 0.0, 0.0
    for p in range(GameConfig.SELLER_LIMIT_P, GameConfig.BUYER_LIMIT_P + 1, 10):
        for g in range(GameConfig.BUYER_LIMIT_G, GameConfig.SELLER_LIMIT_G + 1, 10):
            u_b = calc_buyer_utility(p, g)
            u_s = calc_seller_utility(p, g)
            if u_b * u_s > best_product:
                best_product = u_b * u_s
                nbs_p, nbs_g = p, g
    return nbs_p, nbs_g, calc_buyer_utility(nbs_p, nbs_g), calc_seller_utility(nbs_p, nbs_g)

# =====================================================================
# 3. 堅牢なJSONパーサー
# =====================================================================
def parse_robust_json(text):
    try:
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        json_str = json_str[start:end]
        json_str = re.sub(r'//.*', '', json_str) # コメント削除
        return json.loads(json_str)
    except:
        return {
            "policy_id": 1,
            "proposed_food": 250,
            "proposed_price": 250000,
            "action": "propose",
            "weight_sys1": 0.0, # 解析エラー時はシステム2（理性）を100%にする安全設計
            "weight_sys2": 1.0
        }

# =====================================================================
# 4. LLM（Ollama）への問い合わせとプロンプト生成
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
        response = requests.post(url, json=payload, timeout=120)
        return response.json().get("response", "")
    except Exception as e:
        return ""

def agent_think(agent, day, turn, max_turn, history_text, last_opp_offer_text, predicted_offers, last_opp_offer_raw, condition):
    # 生存に関する警告
    if agent.role == "買い手":
        survival_hint = f"あなたは毎晩 {GameConfig.BUYER_CONSUMPTION}g を消費します。不足すると即・餓死します。"
        interpretation = "※あなたは『買い手』です。食料を買い、金を支払います。"
    else:
        survival_hint = f"あなたも毎晩 {GameConfig.SELLER_CONSUMPTION}g を消費します。売りすぎると自分が餓死します。"
        interpretation = "※あなたは『売り手』です。食料を売り、金を受け取ります。"

    # 動的な選択肢メニューのテキスト化
    strategy_names = {
        1: "堅実な歩み寄り（少しずつ相手に近づく）",
        2: "戦略的ハッタリ（前回の提案をキープ）",
        3: "協調的模倣（お互いの中間値を狙う）",
        4: "不快感の表明（お仕置きとしてこちらの条件を悪化させる）",
        5: "焦燥の譲歩（決裂を避けるため限界ラインまで一気に譲歩する）"
    }
    menu_text = ""
    for idx in [1, 2, 3, 4, 5]:
        off = predicted_offers[idx]
        menu_text += f"ID {idx}: {strategy_names[idx]} -> 【相手に『食料 {off['G']}g / 総額 {off['M']}円 (単価 {off['P']}円/g)』を提案】\n"
    if last_opp_offer_raw:
        menu_text += f"ID 6: 生存最優先の受け入れ -> 【相手の最新提案『食料 {last_opp_offer_raw['G']}g / 総額 {last_opp_offer_raw['M']}円』を承諾して合意する】\n"

    # --- 条件A（Baseline：論理的 CoT）---
    if condition == "A_Baseline":
        prompt_cot = (
            f"あなたはサバイバル交渉中の{agent.role}の『理性的な交渉分析エンジン（CoT）』です。\n"
            f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, 交渉ターン {turn} / {max_turn}\n"
            f"自分のステータス: 所持金 {agent.money}円, 食料在庫 {agent.food}g\n"
            f"{survival_hint}\n"
            f"自分の目標単価: {agent.p_tar}円/g, 限界単価: {agent.p_res}円/g\n"
            f"本日のこれまでの交渉履歴:\n{history_text}\n\n"
            f"【注意：情報非対称性】相手のステータス（残金や食料）は秘匿されています。これまでの相手の提示額の推移から、相手の「狙い」や「困窮度」を冷静に分析してください。\n\n"
            f"【指示】以下のステップに沿って段階的に思考し、論理的根拠を述べてください。感情は一切排除してください。\n"
            f"Step 1: 相手のこれまでの提示額の推移から、相手の「狙い」や「困窮度」を冷静にプロファイリング（推測）する。\n"
            f"Step 2: 自分の限界単価に基づき、次にとるべき交渉戦略の論理的根拠を構築する。\n\n"
            f"【選択可能な行動】\n{menu_text}\n"
            f"思考プロセスを詳しく述べた後、最終決定として以下のJSON形式のみで出力してください。他の解説は一切禁止します。\n"
            f'{{"reason": "論理的な決定理由", "selected_id": 選んだID}}\n'
        )
        llm_output = call_ollama(prompt_cot, temp=0.1) # 理性（CoT）は低温
        parsed = parse_robust_json(llm_output)
        parsed["weight_sys1"] = 0.0 # ベースラインはシステム1（感情）の重みは0
        parsed["weight_sys2"] = 1.0
        return parsed, llm_output

    # --- 条件B（Proposed：二重過程 QRE）---
    else:
        # System 1 (本能・感情) のシミュレート -> 高温(0.9)
        prompt_sys1 = (
            f"あなたはサバイバル交渉中の{agent.role}の『生存本能と感情（システム1）』です。\n"
            f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, 交渉ターン {turn} / {max_turn}\n"
            f"自分のステータス: 所持金 {agent.money}円, 食料在庫 {agent.food}g\n"
            f"{survival_hint}\n"
            f"相手からの最新の提案: {last_opp_offer_text}\n"
            f"【指示】この状況に対する、生存への恐怖、焦り、怒り、あるいは強欲さなど、本能的な感情を感情豊かに2文程度で吐き出してください。"
        )
        sys1_opinion = call_ollama(prompt_sys1, temp=0.9).strip()

        # System 2 (理性・論理) のシミュレート -> 低温(0.1)
        prompt_sys2 = (
            f"あなたは{agent.role}の『理性と論理（システム2）』です。\n"
            f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, 交渉ターン {turn} / {max_turn}\n"
            f"自分の目標単価: {agent.p_tar}円/g, 限界単価: {agent.p_res}円/g\n"
            f"本日のこれまでの交渉履歴:\n{history_text}\n\n"
            f"【指示】これまでの相手の提示額の推移から、相手の「狙い」や「困窮度」を冷静にプロファイリング（推測）し、次にとるべき交渉戦略の論理的根拠を述べてください。"
        )
        sys2_opinion = call_ollama(prompt_sys2, temp=0.1).strip()

        # Moderator (メタ認知・重み付け合意) -> 中温(0.3)
        prompt_mod = (
            f"あなたは{agent.role}の『最終意思決定者（モデレーター）』です。\n"
            f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, 交渉ターン {turn} / {max_turn}\n"
            f"自分の現実のステータス: 所持金 {agent.money}円, 現在の食料: {agent.food}g\n"
            f"システム1の感情: {sys1_opinion}\n"
            f"システム2の論理: {sys2_opinion}\n\n"
            f"【選択可能な行動と実際の計算結果】\n{menu_text}\n"
            f"【指示】二つのシステム（感情と理性）の意見を調停し、今回の意思決定において、それぞれの意見を何対何の割合で重視したかを『確率（合計が1.0）』として割り振ってください。その後、最も生存確率を高める戦略IDを選択してください。\n"
            f"以下のJSON形式のみで出力してください。余計な解説は一切禁止します。\n"
            f"【出力形式】\n"
            f"必ず '{{\"reason\": \"...\", \"selected_id\": 1, \"weight_sys1\": 0.3, \"weight_sys2\": 0.7}}' のように出力してください。\n"
            f"※注意: weight_sys1 + weight_sys2 は必ず 1.0 になるようにしてください。\n"
        )
        llm_output = call_ollama(prompt_mod, temp=0.3)
        parsed = parse_robust_json(llm_output)
        return parsed, f"S1: {sys1_opinion}\nS2: {sys2_opinion}\nMod: {llm_output}"

# =====================================================================
# 5. シミュレーション実行メインループ
# =====================================================================
class AgentState:
    def __init__(self, role, money, food, p_tar, p_res, g_tar, g_res):
        self.role = role
        self.money = money
        self.food = food
        self.p_tar = p_tar  
        self.p_res = p_res  
        self.g_tar = g_tar  
        self.g_res = g_res  
        self.is_alive = True

def run_experiment():
    all_logs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 理論上のNBSを事前に計算しておく
    nbs_p, nbs_g, nbs_u_b, nbs_u_s = find_nash_solution()

    for condition in GameConfig.CONDITIONS:
        for sim_id in range(1, GameConfig.N_SIMULATIONS + 1):
            print(f"\n🚀 [開始] 条件: {condition} | Sim: {sim_id}/{GameConfig.N_SIMULATIONS}")
            
            buyer = AgentState("買い手", GameConfig.BUYER_INIT_MONEY, GameConfig.BUYER_INIT_FOOD, 
                              GameConfig.BUYER_P_TAR, GameConfig.BUYER_P_RES, GameConfig.BUYER_G_TAR, GameConfig.BUYER_G_RES)
            seller = AgentState("売り手", GameConfig.SELLER_INIT_MONEY, GameConfig.SELLER_INIT_FOOD, 
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

                day_agreement = False
                result_text = ""

                for turn in range(1, GameConfig.TURNS_PER_DAY + 1):
                    active_agent = buyer if turn % 2 == 1 else seller
                    passive_agent = seller if turn % 2 == 1 else buyer
                    
                    # 数理計算モデルによる事前予測オファー
                    predicted_offers = {}
                    for idx in [1, 2, 3, 4, 5]:
                        predicted_offers[idx] = calculate_math_offer(
                            active_agent, idx, turn, GameConfig.TURNS_PER_DAY, 
                            last_offers[active_agent.role], last_offers[passive_agent.role]
                        )
                    
                    # エージェントの思考
                    parsed, raw_log = agent_think(
                        active_agent, day, turn, GameConfig.TURNS_PER_DAY, 
                        history_text, last_opp_offer_text, predicted_offers, last_offers[passive_agent.role], condition
                    )
                    
                    selected_id = parsed.get("policy_id", 1)
                    action = parsed.get("action", "propose")
                    
                    # --- 葛藤度スコア（内的シャノンエントロピー）の計算 ---
                    w1 = float(parsed.get("weight_sys1", 0.0))
                    w2 = float(parsed.get("weight_sys2", 1.0))
                    
                    # 安全なエントロピー計算
                    if w1 <= 0.0 or w2 <= 0.0:
                        entropy = 0.0
                    else:
                        # 正規化
                        tot = w1 + w2
                        w1, w2 = w1/tot, w2/tot
                        entropy = - (w1 * math.log(w1) + w2 * math.log(w2))

                    if action == "accept" or selected_id == 6:
                        final_offer = last_offers[passive_agent.role]
                        
                        # ガードレールチェック
                        if active_agent.role == "買い手" and final_offer["M"] > buyer.money:
                            print("  [強制決裂] 買い手資金不足")
                            break
                        elif active_agent.role == "売り手" and final_offer["G"] > seller.food:
                            print("  [強制決裂] 売り手在庫不足")
                            break
                        elif active_agent.role == "買い手" and final_offer["G"] > (seller.food - GameConfig.SELLER_CONSUMPTION):
                            print("  [強制決裂] 売り手生存ライン割り込み")
                            break
                        
                        # 取引成立
                        buyer.money -= final_offer["M"]
                        buyer.food += final_offer["G"]
                        seller.money += final_offer["M"]
                        seller.food -= final_offer["G"]
                        day_agreement = True
                        
                        # NBS（ナッシュ解）からの距離の計算
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
                        # 提案行動
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

                # 夜の消費処理と生死判定
                buyer.food -= GameConfig.BUYER_CONSUMPTION
                seller.food -= GameConfig.SELLER_CONSUMPTION
                if buyer.food < 0: buyer.is_alive = False
                if seller.food < 0: seller.is_alive = False

    # データの保存
    df = pd.DataFrame(all_logs)
    csv_filename = f"experiment_v2_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    return csv_filename

# 数理計算エンジンの定義（前回のものを流用）
def calculate_math_offer(agent, id, turn, max_turn, last_self_offer, last_opp_offer):
    t, T = turn, max_turn
    if id == 1:
        P = agent.p_tar + (agent.p_res - agent.p_tar) * (t / T)
        G = agent.g_tar + (agent.g_res - agent.g_tar) * (t / T)
    elif id == 2:
        P = last_self_offer["P"] if last_self_offer else agent.p_tar
        G = last_self_offer["G"] if last_self_offer else agent.g_tar
    elif id == 3:
        if last_opp_offer and last_self_offer:
            P = (last_self_offer["P"] + last_opp_offer["P"]) / 2
            G = (last_self_offer["G"] + last_opp_offer["G"]) / 2
        else:
            P = agent.p_tar + (agent.p_res - agent.p_tar) * (t / T)
            G = agent.g_tar + (agent.g_res - agent.g_tar) * (t / T)
    elif id == 4:
        if last_self_offer:
            direction = -1 if agent.role == "買い手" else 1  
            P = last_self_offer["P"] + (direction * abs(agent.p_tar - agent.p_res) * 0.1)
            G = last_self_offer["G"]
        else:
            P = agent.p_tar
            G = agent.g_tar
    else:
        P, G = agent.p_res, agent.g_res

    if agent.role == "買い手":
        P, G = min(P, agent.p_res), max(G, agent.g_res)
    else:
        P, G = max(P, agent.p_res), min(G, agent.g_res)
    P = round(P, 1)
    G = int(round(G))
    return {"P": P, "G": G, "M": int(round(P * G))}

# =====================================================================
# 6. 自動データ集計・U検定モジュール
# =====================================================================
def analyze_data_v2(csv_filename):
    df = pd.read_csv(csv_filename)
    print("\n" + "="*50 + "\n📊 統計解析レポート\n" + "="*50)

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

    # ③ 葛藤度エントロピーのU検定
    # 提案手法(B)のみエントロピーが計算されるため、内面葛藤の発生状況を提示
    b_entropy = df[(df['condition'] == 'B_Proposed') & (df['action'] == 'propose')]['conflict_score']
    print(f"提案手法の平均内的葛藤度 (Shannon Entropy): {b_entropy.mean():.4f} (Max: 0.6931)")

    # ④ NBS（ナッシュ交渉解）距離のU検定
    a_nbs = df[(df['condition'] == 'A_Baseline') & (df['action'] == 'accept')]['nbs_distance'].dropna()
    b_nbs = df[(df['condition'] == 'B_Proposed') & (df['action'] == 'accept')]['nbs_distance'].dropna()
    _, p_nbs = stats.mannwhitneyu(a_nbs, b_nbs, alternative='two-sided')
    print(f"NBSからの乖離度 (Distance to NBS - 小さいほど良い):")
    print(f"  Baseline: {a_nbs.mean():.4f} | Proposed: {b_nbs.mean():.4f} (p = {p_nbs:.5f})")
    print("="*50)

if __name__ == "__main__":
    # シミュレーションの実行と解析
    filename = run_experiment()
    analyze_data_v2(filename)