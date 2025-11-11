import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# -----------------------------
# モデル読み込み
# -----------------------------
MODEL_PATH = "model/skin_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["正常", "異常"]


# ラベルマップ読み込み
with open("model/label_map.json", "r") as f:
    class_indices = json.load(f)


# 略号 → 日本語マッピング
label_jp = {
    "nv": "正常／ほくろ",
    "mel": "メラノーマ",
    "bkl": "良性角化症",
    "bcc": "基底細胞がん",
    "akiec": "光線角化症",
    "vasc": "血管腫",
    "df": "皮膚線維腫"
}
    
# class_indices は {'nv':0, 'mel':1, ...} という dict
# 数値 → ラベル に変換する辞書を作る
idx_to_label = {v: label_jp[k] for k, v in class_indices.items()}

# -----------------------------
# ページタイトル
# -----------------------------
st.set_page_config(page_title="AI皮膚チェック（デモ）", page_icon="📸", layout="centered")

st.title("📸 AI皮膚チェック（デモ）")
st.write("皮膚の写真をアップロードすると、AIが簡易診断します。")

# -----------------------------
# ファイルアップロードUI
# -----------------------------
uploaded_file = st.file_uploader(
    "画像ファイルをアップロード",
    type=["jpg", "jpeg", "png"],
)

# 💬 CSSでデザイン調整
st.markdown(
    """
    <style>
    /* ページ背景 */
    body {
        background-color: #f7f9fc;
    }
    /* 結果カード */
    .result-card {
        background-color: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
        transition: transform 0.2s ease-in-out;
    }
    .result-card:hover {
        transform: scale(1.02);
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
    }
    .result-value {
        font-size: 1.1rem;
        color: #4A90E2;
        font-weight: 600;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 推論処理
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # RGBA → RGB変換
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="アップロードされた画像", use_container_width=True)

    # 画像前処理
    input_shape = model.input_shape[1:3]  # (高さ, 幅)
    img_resized = image.resize(input_shape)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 推論
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = idx_to_label[predicted_class]  # idx_to_label は JSON から作成済み
    confidence = prediction[0][predicted_class] * 100

    # -----------------------------
    # 結果カードをHTMLで表示
    # -----------------------------
    result_html = f"""
    <div class="result-card">
        <div class="result-title">🧠 AI診断結果</div>
        <div class="result-value">🔍 判定：{predicted_label}</div>
        <div class="result-value">📊 信頼度：{confidence:.2f}%</div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)

    st.info("※この結果はデモです。実際の診断は医師にご相談ください。")
