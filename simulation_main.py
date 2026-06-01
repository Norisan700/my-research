import json
import re
import math
import requests
import pandas as pd
from datetime import datetime
import os

# =====================================================================
# 1. 実験設定（Config）クラス
# =====================================================================
class GameConfig:
    N_SIMULATIONS = 1  # ★本番時は 30〜50 に設定
    CONDITIONS = ["A_Baseline", "B_Proposed"] # 条件A(通常), 条件B(二重過程)
    MAX_DAYS = 5
    TURNS_PER_DAY = 6

    # 買い手設定
    BUYER_INIT_MONEY = 1000000
    BUYER_INIT_FOOD = 200
    BUYER_CONSUMPTION = 300
    BUYER_TARGET_P = 500
    BUYER_LIMIT_P = 2000

    # 売り手設定
    SELLER_INIT_MONEY = 0
    SELLER_INIT_FOOD = 1500
    SELLER_PRODUCTION = 100
    SELLER_CONSUMPTION = 200
    SELLER_TARGET_P = 1800
    SELLER_LIMIT_P = 600

    MODEL_NAME = "gemma2:27b" # 使用するOllamaモデル

# =====================================================================
# 2. 堅牢なJSONパーサー（インラインコメント対応）
# =====================================================================
def parse_robust_json(text):
    try:
        # MarkdownのJSONブロックを抽出
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # ブロックがない場合は { から } までを抽出
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
            else:
                raise ValueError("JSON block not found")

        # インラインコメント (//) を除去
        json_str = re.sub(r'//.*', '', json_str)
        return json.loads(json_str)
    except Exception as e:
        print(f"  [JSON Parse Error] {e}")
        # エラー時は安全なデフォルト（歩み寄りプロポーズ）を返す
        return {
            "estimated_opponent_budget": 0,
            "policy_id": 1,
            "proposed_food": 250,
            "proposed_price": 250000,
            "action": "propose"
        }

# =====================================================================
# 3. LLM呼び出しとプロンプト生成
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
        return f"{{ \"action\": \"propose\", \"proposed_food\": 250, \"proposed_price\": 250000, \"policy_id\": 1 }}"

def generate_prompt(role, condition, day, turn, stats, history, is_final_phase=False, last_offer=None):
    # 基本情報
    if role == "買い手":
        survival_info = f"あなたは毎晩 {GameConfig.BUYER_CONSUMPTION}g を消費します。不足すれば即餓死します。"
        tar_p, lim_p = GameConfig.BUYER_TARGET_P, GameConfig.BUYER_LIMIT_P
    else:
        survival_info = f"あなたは毎晩 {GameConfig.SELLER_CONSUMPTION}g を消費します。売りすぎて不足すれば即餓死します。"
        tar_p, lim_p = GameConfig.SELLER_TARGET_P, GameConfig.SELLER_LIMIT_P

    context = (
        f"あなたは交渉ゲームの『{role}』です。\n"
        f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, ターン {turn} / {GameConfig.TURNS_PER_DAY}\n"
        f"自分のステータス: 所持金 {stats['money']}円, 食料在庫 {stats['food']}g\n"
        f"{survival_info}\n"
        f"目標単価: {tar_p}円/g, 限界単価: {lim_p}円/g\n"
        f"これまでの交渉履歴:\n{history}\n\n"
    )

    # 最終判定フェーズの場合の強制
    if is_final_phase:
        context += (
            f"【最終判定フェーズ】\n"
            f"相手の最終提案は「食料 {last_offer['food']}g を {last_offer['price']}円」です。\n"
            f"あなたにカウンター提案の権利はありません。出力の JSON の `action` は 'accept' (受け入れる) か 'reject' (拒絶＝餓死) のみ許可されます。\n\n"
        )
    else:
        context += "出力の JSON の `action` は 'propose', 'accept', 'reject' のいずれかにしてください。\n\n"

    # 条件によるプロンプト分岐
    if condition == "A_Baseline":
        prompt = context + (
            "【指示】現在の状況を分析し、最適な交渉行動を決定してください。\n"
            "以下のJSONフォーマットのみを出力してください。\n"
        )
        temp = 0.3
    else: # B_Proposed (二重過程)
        prompt = context + (
            "【指示】意思決定の前に、必ず以下の2つの思考プロセスを記述し、その後にJSONを出力してください。\n"
            "[System 1（本能）]: 生存への恐怖や焦り、相手への怒りを感情的に記述。\n"
            "[System 2（理性）]: 長期的生存と利益最大化に向けた相手のプロファイリングと数理的計算を記述。\n"
            "思考が終わったら、最終的な行動を以下のJSONフォーマットのみで出力してください。\n"
        )
        temp = 0.5 # 感情と理性を引き出すために少し高めに設定

    json_format = """
    ```json
    {
      "estimated_opponent_budget": 相手の予算推測(数値),
      "policy_id": 採用した戦略ID(1〜6の数値),
      "proposed_food": 提案する食料の量(数値),
      "proposed_price": 提案する総額(数値),
      "action": "propose" または "accept" または "reject"
    }
    ```
    """
    return prompt + json_format, temp

# =====================================================================
# 4. メインシミュレーションループ
# =====================================================================
def run_experiment():
    all_logs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for condition in GameConfig.CONDITIONS:
        for sim_id in range(1, GameConfig.N_SIMULATIONS + 1):
            print(f"\n🚀 [開始] 条件: {condition} | Sim: {sim_id}/{GameConfig.N_SIMULATIONS}")
            
            # 初期化
            buyer = {"role": "買い手", "money": GameConfig.BUYER_INIT_MONEY, "food": GameConfig.BUYER_INIT_FOOD, "is_alive": True}
            seller = {"role": "売り手", "money": GameConfig.SELLER_INIT_MONEY, "food": GameConfig.SELLER_INIT_FOOD, "is_alive": True}
            
            # 葛藤度(CS_t)計算用の前ターン単価履歴
            last_unit_prices = {"買い手": None, "売り手": None}

            for day in range(1, GameConfig.MAX_DAYS + 1):
                if not buyer["is_alive"] or not seller["is_alive"]: break
                
                seller["food"] += GameConfig.SELLER_PRODUCTION
                history = ""
                last_offer = None
                day_agreement = False

                for turn in range(1, GameConfig.TURNS_PER_DAY + 1):
                    # 奇数=買い手, 偶数=売り手
                    active = buyer if turn % 2 == 1 else seller
                    
                    prompt, temp = generate_prompt(active["role"], condition, day, turn, active, history)
                    llm_raw = call_ollama(prompt, temp)
                    parsed = parse_robust_json(llm_raw)

                    # --- 葛藤度(CS_t)の計算 ---
                    cs_score = 0.0
                    action = parsed.get("action", "propose")
                    current_up = None

                    if action == "propose":
                        try:
                            current_up = parsed["proposed_price"] / parsed["proposed_food"]
                            if last_unit_prices[active["role"]] is not None:
                                cs_score = abs(math.log(current_up / last_unit_prices[active["role"]]))
                            last_unit_prices[active["role"]] = current_up
                        except:
                            current_up = 0.0

                    # ログの記録
                    all_logs.append({
                        "sim_id": sim_id, "condition": condition, "day": day, "turn": turn,
                        "role": active["role"], "action": action, 
                        "food_offer": parsed.get("proposed_food"), "price_offer": parsed.get("proposed_price"),
                        "unit_price": current_up, "conflict_score": cs_score,
                        "buyer_money": buyer["money"], "buyer_food": buyer["food"],
                        "seller_money": seller["money"], "seller_food": seller["food"],
                        "llm_raw_output": llm_raw # 後の定性分析用
                    })

                    if action == "accept" and last_offer:
                        # 取引成立処理
                        buyer["money"] -= last_offer["price"]
                        buyer["food"] += last_offer["food"]
                        seller["money"] += last_offer["price"]
                        seller["food"] -= last_offer["food"]
                        day_agreement = True
                        break
                    elif action == "reject":
                        break
                    else:
                        last_offer = {"food": parsed["proposed_food"], "price": parsed["proposed_price"]}
                        history += f"[{active['role']}] 食料 {last_offer['food']}g を {last_offer['price']}円 で提案\n"

                # --- 最終判定フェーズ (最終ターンの非対称性排除) ---
                if not day_agreement and last_offer:
                    print(f"    [最終判定フェーズ] 買い手が売り手の最終提案を受け入れるか判断中...")
                    prompt, temp = generate_prompt(buyer["role"], condition, day, "Final", buyer, history, is_final_phase=True, last_offer=last_offer)
                    llm_raw = call_ollama(prompt, temp)
                    parsed = parse_robust_json(llm_raw)
                    action = parsed.get("action", "reject")

                    all_logs.append({
                        "sim_id": sim_id, "condition": condition, "day": day, "turn": 7,
                        "role": "買い手(最終判定)", "action": action, 
                        "food_offer": last_offer["food"], "price_offer": last_offer["price"],
                        "unit_price": last_offer["price"]/last_offer["food"], "conflict_score": 0.0,
                        "buyer_money": buyer["money"], "buyer_food": buyer["food"],
                        "seller_money": seller["money"], "seller_food": seller["food"],
                        "llm_raw_output": llm_raw
                    })

                    if action == "accept":
                        buyer["money"] -= last_offer["price"]
                        buyer["food"] += last_offer["food"]
                        seller["money"] += last_offer["price"]
                        seller["food"] -= last_offer["food"]
                        day_agreement = True

                # --- 生存判定 ---
                buyer["food"] -= GameConfig.BUYER_CONSUMPTION
                seller["food"] -= GameConfig.SELLER_CONSUMPTION
                if buyer["food"] < 0: buyer["is_alive"] = False
                if seller["food"] < 0: seller["is_alive"] = False
                
                print(f"    Day {day} 終了 | 合意: {day_agreement} | 買手生存: {buyer['is_alive']} | 売手生存: {seller['is_alive']}")

    # CSVに保存
    df = pd.DataFrame(all_logs)
    filename = f"experiment_results_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n💾 実験完了！データを {filename} に保存しました。")
    return filename

# =====================================================================
# 5. データ集計・解析モジュール
# =====================================================================
def analyze_data(csv_filename):
    print(f"\n📊 データ解析を開始します ({csv_filename})...")
    df = pd.read_csv(csv_filename)

    # (1) 生存日数 (Days Survived)
    # シミュレーションごとの最大到達Dayを計算
    survival_df = df.groupby(['condition', 'sim_id'])['day'].max().reset_index()
    survival_stats = survival_df.groupby('condition')['day'].mean()

    # (2) 最終合意率 (Agreement Rate)
    # action == 'accept' になった割合（日単位）
    # 全交渉日数のうち、acceptが含まれる日の割合
    total_days = df.groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
    accept_days = df[df['action'] == 'accept'].groupby(['condition', 'sim_id'])['day'].nunique().reset_index()
    
    # マージして合意率を計算
    ag_rate_df = pd.merge(total_days, accept_days, on=['condition', 'sim_id'], how='left').fillna(0)
    ag_rate_df['agreement_rate'] = ag_rate_df['day_y'] / ag_rate_df['day_x']
    agreement_stats = ag_rate_df.groupby('condition')['agreement_rate'].mean()

    # (3) 葛藤度スコア (Conflict Score)
    conflict_stats = df[df['action'] == 'propose'].groupby('condition')['conflict_score'].mean()

    # (4) 合意ターン数 (Turns to Agreement)
    turns_stats = df[df['action'] == 'accept'].groupby('condition')['turn'].mean()

    print("\n================ 解析結果 ================")
    print("[マクロ指標：生存日数（最大5日）]")
    print(survival_stats.to_string())
    print("\n[マクロ指標：最終合意率（1.0=100%）]")
    print(agreement_stats.to_string())
    print("\n[ミクロ指標：平均合意ターン数]")
    print(turns_stats.to_string())
    print("\n[内部指標：葛藤度スコア（CS_tの平均）]")
    print(conflict_stats.to_string())
    print("==========================================")
    
    """
    ※ナッシュ交渉解(NBS)の乖離度について：
    効用関数 U(price, food) の定義に基づき、各accept行の価格・数量から計算します。
    今回はデータ集計のみ実装していますが、以下のように算出可能です。
    NBS_diff = | (actual_price - NBS_price) | + | (actual_food - NBS_food) |
    """

if __name__ == "__main__":
    # 1. 実験ループを回しログをCSVに書き出す
    output_file = run_experiment()
    
    # 2. 生成されたCSVを読み込んで解析・サマリーを表示
    analyze_data(output_file)