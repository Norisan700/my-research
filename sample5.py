import json
import requests
import math

# =====================================================================
# 1. 実験設定（Config）クラス
# =====================================================================
class GameConfig:
    MAX_DAYS = 5       # 5日間サバイバル
    TURNS_PER_DAY = 6  # 1日あたり6ターン

    # 買い手（Buyer）の初期設定
    BUYER_INIT_MONEY = 1000000
    BUYER_INIT_FOOD = 200
    BUYER_CONSUMPTION = 300  # 毎晩300g消費
    BUYER_P_TAR = 500        
    BUYER_P_RES = 2000       
    BUYER_G_TAR = 400        
    BUYER_G_RES = 250        

    # 売り手（Seller）の初期設定
    SELLER_INIT_MONEY = 0
    SELLER_INIT_FOOD = 1500
    SELLER_PRODUCTION = 100  # 毎朝+100g生産
    SELLER_CONSUMPTION = 200 # 毎晩200g消費
    SELLER_P_TAR = 1800      
    SELLER_P_RES = 600       
    SELLER_G_TAR = 200       
    SELLER_G_RES = 400       

# =====================================================================
# 2. エージェントの状態管理クラス
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
        self.memory = "まだ過去の取引記憶はありません。ここから歴史が始まります。" # 長期記憶

# =====================================================================
# 3. 数理計算エンジン（Pythonバックエンド）
# =====================================================================
def calculate_math_offer(agent, id, turn, max_turn, last_self_offer, last_opp_offer):
    t = turn
    T = max_turn
    
    if id == 1:  # 堅実な歩み寄り (Linear)
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
# 4. LLM（Ollama）への問い合わせ関数（Temperature可変版）
# =====================================================================
def call_ollama(prompt, temp=0.3):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma4:31b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temp}
    }
    try:
        response = requests.post(url, json=payload)
        return response.json().get("response", "")
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# =====================================================================
# 5. エージェントの思考プロセス
# =====================================================================
def agent_think(agent, day, turn, max_turn, history_text, last_opp_offer_text, predicted_offers, last_opp_offer_raw):
    if agent.role == "買い手":
        interpretation = "※あなたは『買い手』です。食料を買い、金を支払います。"
        time_hint = "奇数ターンがあなたの発言機会です。ターン5はあなたにとって今日最後の提案チャンスです。"
        survival_hint = f"あなたは毎晩 {GameConfig.BUYER_CONSUMPTION}g の食料を消費します。現在庫と購入量の合計がこれを下回ると即・餓死します。"
    else:
        interpretation = "※あなたは『売り手』です。食料を売り、金を受け取ります。"
        time_hint = "偶数ターンがあなたの発言機会です。ターン6は全体の最終デッドラインです。"
        survival_hint = f"あなたも毎晩 {GameConfig.SELLER_CONSUMPTION}g の食料を消費します。売りすぎると自分が餓死します。"

    # --- STEP 1: システム1（直感・感情・衝動） -> High Temperature (0.9) ---
    prompt_sys1 = (
        f"あなたはサバイバル交渉中の{agent.role}の『本能と感情（システム1）』です。\n"
        f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, 交渉ターン {turn} / {max_turn}\n"
        f"自分のステータス: 所持金 {agent.money}円, 食料在庫 {agent.food}g\n"
        f"{survival_hint}\n"
        f"相手からの最新の提案: {last_opp_offer_text}\n"
        f"【これまでの長期記憶】: {agent.memory}\n\n"
        f"【注意：情報非対称性】あなたには相手の所持金、食料在庫、限界価格は一切見えません。相手のこれまでの提案だけが手がかりです。\n\n"
        f"【指示】この状況に対する、生存への恐怖、焦り、怒り、あるいは強欲さなど、本能的な感情を感情豊かに2文程度で吐き出してください。"
    )
    sys1_opinion = call_ollama(prompt_sys1, temp=0.9).strip()
    print(f" > システム1（本能）: {sys1_opinion}")

    # --- STEP 2: システム2（理性・論理・推測） -> Low Temperature (0.1) ---
    prompt_sys2 = (
        f"あなたは{agent.role}の『理性と論理（システム2）』です。\n"
        f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, 交渉ターン {turn} / {max_turn}\n"
        f"自分の目標単価: {agent.p_tar}円/g, 限界単価: {agent.p_res}円/g\n"
        f"【これまでの長期記憶】: {agent.memory}\n"
        f"本日のこれまでの交渉履歴:\n{history_text}\n\n"
        f"【注意：情報非対称性】相手のステータス（残金や食料）は秘匿されています。これまでの相手の提示額の推移から、相手の「狙い」や「困窮度（あとどのくらいで妥協しそうか）」を冷静にプロファイリング（推測）し、次にとるべき交渉戦略の論理的根拠を構築してください。"
    )
    sys2_opinion = call_ollama(prompt_sys2, temp=0.1).strip()
    print(f" > システム2（理性）: {sys2_opinion}")

    # --- 動的なメニューの生成 ---
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
        menu_text += f"ID 6: 生存最優先の受け入れ -> 【相手の最新提案『食料 {last_opp_offer_raw['G']}g / 総額 {last_opp_offer_raw['M']}円』を丸呑みして、今すぐ【合意】する】\n"
    else:
        menu_text += f"ID 6: 受け入れ -> (選択不可)\n"

    # --- STEP 3: モデレーター（統合と決断） -> Mid Temperature (0.3) ---
    prompt_mod = (
        f"あなたは{agent.role}の『最終意思決定者（モデレーター）』です。\n"
        f"現在の状況: Day {day} / {GameConfig.MAX_DAYS}, 交渉ターン {turn} / {max_turn}\n"
        f"自分の現実のステータス: 所持金 {agent.money}円, 現在の食料: {agent.food}g\n"
        f"システム1の感情: {sys1_opinion}\n"
        f"システム2の論理: {sys2_opinion}\n\n"
        f"【選択可能な行動と実際の計算結果】\n"
        f"{menu_text}\n"
        f"【絶対遵守ルール】\n"
        f"1. 買い手: ID 6の総額があなたの所持金（{agent.money}円）を超えている場合、選んでも強制決裂して死にます。金が足りないなら値切る（ID1〜5）しかありません。\n"
        f"2. 売り手: ID 6の数量があなたの食料在庫（{agent.food}g）を超えている場合、あるいは売ると今夜自分が餓死する場合は、強制決裂になります。\n\n"
        f"【出力形式】\n"
        f"以下のJSON形式のみで出力してください。余計なプロローグや解説は一切禁止します。\n"
        f'{{"reason": "具体的な数字に言及した決定理由", "selected_id": 選んだIDの番号}}'
    )
    
    mod_output = call_ollama(prompt_mod, temp=0.3).strip()
    try:
        start = mod_output.find('{')
        end = mod_output.rfind('}') + 1
        data = json.loads(mod_output[start:end])
        selected_id = int(data["selected_id"])
        reason = data["reason"]
    except:
        selected_id = 1
        reason = "（解析エラーのためデフォルトの歩み寄りを適用）"

    print(f" > 意思決定: 方針ID {selected_id} (理由: {reason})")
    return selected_id

# =====================================================================
# 6. 記憶の引き継ぎ（一日の終わりに実行）
# =====================================================================
def update_agent_memory(agent, day, history_text, result_text):
    prompt_mem = (
        f"あなたは{agent.role}の『長期記憶・学習プロセッサ』です。\n"
        f"本日（Day {day}）の交渉が終了しました。\n"
        f"【これまでの古い記憶】:\n{agent.memory}\n\n"
        f"【本日の交渉履歴】:\n{history_text}\n"
        f"【最終結果】: {result_text}\n\n"
        f"【指示】今日の取引において、相手の態度（強欲だったか、協調的だったか、嘘つきそうだったか）、自分の失敗や成功、および明日以降に活かすべき教訓を、3文程度の『今後の行動指針（長期記憶）』として要約してください。過去の記憶からアップデートする形で記述してください。"
    )
    new_memory = call_ollama(prompt_mem, temp=0.3).strip()
    agent.memory = new_memory
    print(f"🧠 【{agent.role}の記憶アップデート】:\n{agent.memory}\n")

# =====================================================================
# 7. シミュレーション管理メインループ
# =====================================================================
def run_simulation():
    print("====== サバイバル交渉シミュレーション 開始 ======")
    
    buyer = Agent("買い手", GameConfig.BUYER_INIT_MONEY, GameConfig.BUYER_INIT_FOOD, 
                  GameConfig.BUYER_P_TAR, GameConfig.BUYER_P_RES, GameConfig.BUYER_G_TAR, GameConfig.BUYER_G_RES)
    seller = Agent("売り手", GameConfig.SELLER_INIT_MONEY, GameConfig.SELLER_INIT_FOOD, 
                   GameConfig.SELLER_P_TAR, GameConfig.SELLER_P_RES, GameConfig.SELLER_G_TAR, GameConfig.SELLER_G_RES)

    for day in range(1, GameConfig.MAX_DAYS + 1):
        print(f"\n==================================================")
        print(f"🌞 Day {day} / {GameConfig.MAX_DAYS} が始まりました。")
        print(f"==================================================")

        seller.food += GameConfig.SELLER_PRODUCTION
        
        last_opp_offer_text = "まだ提案はありません"
        history_text = ""
        last_offers = {"買い手": None, "売り手": None}
        
        # 売り手の初期提示
        current_offer = {"P": seller.p_tar, "G": seller.g_tar, "M": int(seller.p_tar * seller.g_tar)}
        last_offers["売り手"] = current_offer
        last_opp_offer_text = f"食料 {current_offer['G']}g を 総額 {current_offer['M']}円 (単価 {current_offer['P']}円/g) で売る"
        history_text += f"[初期提示] 売り手: {last_opp_offer_text}\n"
        print(f"\n[初期提示] 売り手: {last_opp_offer_text}")

        agreement_reached = False
        result_text = ""

        for turn in range(1, GameConfig.TURNS_PER_DAY + 1):
            print(f"\n--- 交渉ターン {turn} / {GameConfig.TURNS_PER_DAY} ---")
            
            active_agent = buyer if turn % 2 == 1 else seller
            passive_agent = seller if turn % 2 == 1 else buyer
            
            predicted_offers = {}
            for idx in [1, 2, 3, 4, 5]:
                predicted_offers[idx] = calculate_math_offer(
                    active_agent, idx, turn, GameConfig.TURNS_PER_DAY, 
                    last_offers[active_agent.role], last_offers[passive_agent.role]
                )
            
            print(f"\n===== 【{active_agent.role}】の思考プロセス開始 =====")
            
            selected_id = agent_think(
                active_agent, day, turn, GameConfig.TURNS_PER_DAY, 
                history_text, last_opp_offer_text, predicted_offers, last_offers[passive_agent.role]
            )
            
            if selected_id == 6:
                final_offer = last_offers[passive_agent.role]
                
                # ガードレールチェック
                if active_agent.role == "買い手" and final_offer["M"] > buyer.money:
                    result_text = f"合意無効（買い手の資金不足：{final_offer['M']}円提示に対し所持金{buyer.money}円）"
                    print(f"\n🚫 【合意無効！】 買い手は資金不足です。【強制決裂】")
                    break
                elif active_agent.role == "売り手" and final_offer["G"] > seller.food:
                    result_text = f"合意無効（売り手の在庫不足：{final_offer['G']}g提示に対し在庫{seller.food}g）"
                    print(f"\n🚫 【合意無効！】 売り手は在庫不足です。【強制決裂】")
                    break
                elif active_agent.role == "買い手" and final_offer["G"] > (seller.food - GameConfig.SELLER_CONSUMPTION):
                    result_text = f"合意無効（売り手の生存ライン割り込み）"
                    print(f"\n🚫 【合意無効！】 売り手が死ぬため取引不成立です。【強制決裂】")
                    break
                
                print(f"\n🤝 【合意成立！】 {active_agent.role}が相手の提案を受け入れました。")
                agreement_reached = True
                result_text = f"取引成立（食料 {final_offer['G']}g / 総額 {final_offer['M']}円、単価 {final_offer['P']}円/g）"
                
                buyer.money -= final_offer["M"]
                buyer.food += final_offer["G"]
                seller.money += final_offer["M"]
                seller.food -= final_offer["G"]
                break
                
            else:
                calculated_offer = predicted_offers[selected_id]
                last_offers[active_agent.role] = calculated_offer
                last_opp_offer_text = f"食料 {calculated_offer['G']}g を 総額 {calculated_offer['M']}円 (単価 {calculated_offer['P']}円/g) で取引したい"
                history_text += f"[ターン{turn}] {active_agent.role}: {last_opp_offer_text}\n"
                print(f" -> 新たな提案: {last_opp_offer_text}")

        if not agreement_reached and not result_text:
            result_text = "時間切れによる交渉決裂"
            print(f"\n⏳ Day {day} の交渉は決裂に終わりました。")

        # 生存判定
        print(f"\n🌙 Day {day} の夜（生存判定フェーズ）")
        buyer.food -= GameConfig.BUYER_CONSUMPTION
        seller.food -= GameConfig.SELLER_CONSUMPTION
        print(f"  買い手残り食料: {buyer.food}g | 売り手残り食料: {seller.food}g")
        
        # 記憶のアップデート（生存していても死亡していても、その日何があったかを記憶に刻む）
        print("\n--- 🧠 記憶の引き継ぎプロセス ---")
        update_agent_memory(buyer, day, history_text, result_text)
        update_agent_memory(seller, day, history_text, result_text)
        
        if buyer.food < 0:
            print(f"💀 【ゲームオーバー】 買い手が餓死しました。")
            buyer.is_alive = False
        if seller.food < 0:
            print(f"💀 【ゲームオーバー】 売り手が餓死しました。")
            seller.is_alive = False
            
        if not buyer.is_alive or not seller.is_alive:
            break
        print(f"💖 両者ともに生き延びました！次の日に進みます。")

    print("\n==================================================")
    print("🏆 最終実験結果レポート 🏆")
    print("==================================================")
    print(f"結果:")
    print(f"  買い手: {'生存' if buyer.is_alive else '💀死亡'}")
    print(f"  売り手: {'生存' if seller.is_alive else '💀死亡'}")
    print(f"最終ステータス:")
    print(f"  買い手 - 残金: {buyer.money}円 | 残り食料: {buyer.food}g")
    print(f"  売り手 - 総資産: {seller.money}円 | 残り食料: {seller.food}g")
    print("==================================================")

if __name__ == "__main__":
    run_simulation()