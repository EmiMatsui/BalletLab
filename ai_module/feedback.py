import openai

def _cue_for_joint(joint, delta):
    if abs(delta) < 5: return None
    if "knee" in joint: return "膝の向きを意識してまっすぐ伸ばそう"
    if "hip" in joint: return "骨盤を安定させて脚の付け根から動かそう"
    if "elbow" in joint: return "肘を下げずに腕を長くキープ"
    return "体幹をまっすぐ保つ"

def generate_feedback(per_frame, cooldown=8):
    tips = []
    last = None
    cool = 0
    for f in per_frame:
        d = f["diffs"]
        worst = max(d.items(), key=lambda x: abs(x[1]))
        cue = _cue_for_joint(*worst)
        if cue and (cue != last or cool <= 0):
            tips.append(cue)
            last = cue
            cool = cooldown
        else:
            cool = max(0, cool - 1)
    return list(dict.fromkeys(tips))

def generate_chatgpt_feedback(per_frame_results, overall_score, model="gpt-4o-mini"):
    """
    ChatGPTを使って自然言語フィードバックを生成。
    OpenAI APIキーを環境変数OPENAI_API_KEYに設定しておく必要があります。
    """
    summary_text = " ".join([
        f"Frame {i}: Score={r['score']:.1f}, MainDiffs={list(r['diffs'].keys())[:3]}"
        for i, r in enumerate(per_frame_results[::max(1, len(per_frame_results)//20)])  # 約20フレーム分のみ要約
    ])

    prompt = f"""
    あなたはプロのバレエ講師AIです。
    以下の姿勢解析結果をもとに、生徒に対して
    優しく具体的に改善点を指導してください。
    全体スコア: {overall_score:.1f}点
    解析概要: {summary_text}
    """

    try:
        client = openai.OpenAI()  # pip install openai
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは生徒のやる気を引き出すバレエ講師AIです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ ChatGPTフィードバック生成エラー: {e}"
