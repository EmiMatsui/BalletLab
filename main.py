import os
import tempfile
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from ai_module.pose_loader import load_pose_from_video
from ai_module.normalizer import normalize_sequence
from ai_module.aligner import simple_dtw_band
from ai_module.analyzer import analyze_sequences
from ai_module.feedback import generate_feedback, generate_chatgpt_feedback

# .env 読み込み
load_dotenv()

app = FastAPI(title="Ballet AI Analyzer (Modular API)", version="3.0")

# CORS許可（HTMLからアクセス可能にする）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render supplies PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)


# 共通処理: 姿勢データ取得と解析
def process_videos(ideal_video: UploadFile, user_video: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ideal_tmp:
        ideal_path = ideal_tmp.name
        ideal_tmp.write(ideal_video.file.read())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as user_tmp:
        user_path = user_tmp.name
        user_tmp.write(user_video.file.read())

    try:
        ideal_raw = load_pose_from_video(ideal_path)
        user_raw = load_pose_from_video(user_path)
        ideal = normalize_sequence(ideal_raw)
        user = normalize_sequence(user_raw)
        mapping = simple_dtw_band(ideal, user)
        per_frame, overall = analyze_sequences(ideal, user, mapping)
        return ideal, user, mapping, per_frame, overall
    finally:
        os.remove(ideal_path)
        os.remove(user_path)


# ──────────────────────────────────────────────
# 🎯 スコアのみ返すAPI
# ──────────────────────────────────────────────
@app.post("/analyze/score")
async def analyze_score(ideal_video: UploadFile, user_video: UploadFile):
    _, _, _, _, overall = process_videos(ideal_video, user_video)
    return Response(
        content=json.dumps({"score": float(overall)}),
        media_type="application/json"
    )


# ──────────────────────────────────────────────
# 🩰 基本フィードバックのみ返すAPI
# ──────────────────────────────────────────────
@app.post("/analyze/feedback/basic")
async def analyze_basic_feedback(ideal_video: UploadFile, user_video: UploadFile):
    _, _, _, per_frame, _ = process_videos(ideal_video, user_video)
    basic_feedback = generate_feedback(per_frame)
    return Response(
        content=json.dumps({"basic_feedback": [str(x) for x in basic_feedback]}),
        media_type="application/json"
    )


# ──────────────────────────────────────────────
# 💬 ChatGPTフィードバックのみ返すAPI
# ──────────────────────────────────────────────
@app.post("/analyze/feedback/chatgpt")
async def analyze_chatgpt_feedback(ideal_video: UploadFile, user_video: UploadFile):
    _, _, _, per_frame, overall = process_videos(ideal_video, user_video)
    chatgpt_feedback = generate_chatgpt_feedback(per_frame, overall)
    return Response(
        content=json.dumps({"chatgpt_feedback": str(chatgpt_feedback)}),
        media_type="application/json"
    )
