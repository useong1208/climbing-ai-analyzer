import sys
from pathlib import Path
import tempfile

import streamlit as st

# 이 파일(streamlit_app.py)과 같은 폴더를 import 경로에 추가
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from overlay_video import analyze_video

st.set_page_config(page_title="Climbing AI Analyzer", layout="wide")
st.title("🧗 클라이밍 분석기 (영상 업로드 → 오버레이/CSV/TXT 생성)")
st.caption("영상 1개 업로드 → 분석 시작 → 결과 확인/다운로드")

# --- 옵션(간단) ---
st.sidebar.header("옵션")
target_fps = st.sidebar.slider("처리 FPS (낮을수록 빠름)", 0, 30, 15)
resize_width = st.sidebar.slider("리사이즈 가로(px) (0=원본)", 0, 1920, 960, step=10)
model_complexity = st.sidebar.selectbox("정확도(0 빠름 / 2 정확)", [0, 1, 2], index=1)

st.sidebar.divider()
st.sidebar.write("팔다리 인식(민감도) - 기본값 그대로 두면 됨")
min_det_conf = st.sidebar.slider("min_det_conf", 0.10, 0.90, 0.35, 0.05)
min_track_conf = st.sidebar.slider("min_track_conf", 0.10, 0.90, 0.35, 0.05)

st.sidebar.divider()

st.sidebar.divider()
st.sidebar.write("팔다리 인식(민감도) — 기본값 그대로 두면 됨")
min_det_conf = st.sidebar.slider("min_det_conf", 0.10, 0.90, 0.35, 0.05)
min_track_conf = st.sidebar.slider("min_track_conf", 0.10, 0.90, 0.35, 0.05)

uploaded = st.file_uploader("🎥 영상 업로드 (.mp4 / .mov / .avi)", type=["mp4", "mov", "avi"])

if uploaded is None:
    st.info("왼쪽 위에서 영상 파일을 업로드해줘.")
    st.stop()

st.write("업로드됨:", uploaded.name, f"({uploaded.size/1024/1024:.1f} MB)")

run = st.button("🚀 분석 시작", type="primary")

if run:
    with st.spinner("분석 중... (영상 길이에 따라 걸릴 수 있음)"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            input_path = tmp / "input.mp4"
            out_video = tmp / "overlay.mp4"
            out_csv = tmp / "analysis.csv"
            out_txt = tmp / "summary.txt"

            input_path.write_bytes(uploaded.read())

            ret = analyze_video(
                input_path=str(input_path),
                out_video_path=str(out_video),
                out_csv_path=str(out_csv),
                out_txt_path=str(out_txt),
                target_fps=float(target_fps),
                resize_width=int(resize_width),
                model_complexity=int(model_complexity),
                min_det_conf=float(min_det_conf),
                min_track_conf=float(min_track_conf),
            )

            if ret != 0:
                st.error("❌ 분석 실패. (overlay_video.py 내부 오류/경로 문제일 가능성)")
                st.stop()

            st.success("✅ 분석 완료!")

            # 결과 미리보기
            st.subheader("🎬 오버레이 영상")
            st.video(out_video.read_bytes())

            st.subheader("📝 요약 텍스트")
            st.text(out_txt.read_text(encoding="utf-8", errors="ignore"))

            st.subheader("⬇️ 다운로드")
            st.download_button("오버레이 영상 (.mp4) 다운로드", out_video.read_bytes(), file_name="overlay.mp4")
            st.download_button("분석 데이터 (.csv) 다운로드", out_csv.read_bytes(), file_name="analysis.csv")
            st.download_button("요약 (.txt) 다운로드", out_txt.read_bytes(), file_name="summary.txt")
