# frontend/app.py

import streamlit as st
from PIL import Image
import subprocess
import os
from pathlib import Path
import shutil
import re
import time

# --------------------- Paths ---------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "output"
DERAINED_DIR = DATA_DIR / "derained"
DETECTED_DIR = DATA_DIR / "detected"
CROPPED_DIR = DATA_DIR / "cropped"

for folder in [INPUT_DIR, OUTPUT_DIR, DERAINED_DIR, DETECTED_DIR, CROPPED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --------------------- Streamlit Config ---------------------
st.set_page_config(page_title="🌧️ StormSight", page_icon="🌧️", layout="centered")
st.title("🌧️ StormSight: Derain + Text Extraction")
st.markdown("Upload a **rainy image** and we’ll clean it up & pull out the text!")

# --------------------- Utility Functions ---------------------
def clear_all_data():
    for folder in [INPUT_DIR, OUTPUT_DIR, DERAINED_DIR, DETECTED_DIR, CROPPED_DIR]:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
    st.session_state.clear()

def save_uploaded_image(uploaded_file):
    clear_all_data()
    file_path = INPUT_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return str(file_path)

def run_pipeline(input_path):
    pipeline_script = BASE_DIR / "src" / "core" / "pipeline.py"
    cmd = f"python \"{pipeline_script}\""

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        status = st.empty()
        while True:
            return_code = process.poll()
            if return_code is not None:
                break
            status.info("⏳ Processing image... Please wait...")
            time.sleep(2)

        stdout, stderr = process.communicate()

        if return_code != 0:
            st.error(f"Pipeline failed:\n{stderr}")
            return None

        # Get derained image
        output_img = list(DERAINED_DIR.glob("*"))[0] if DERAINED_DIR.exists() else None

        # Get extracted text
        result_txt = OUTPUT_DIR / "results.txt"
        text_out = "No text detected"
        if result_txt.exists():
            with open(result_txt, "r") as f:
                # Extract only the text before ', Confidence:'
                lines = f.read().strip().split('\n')
                text_only = []
                print(lines)
                for line in lines:
                    # Split on ', Confidence:' and take the first part (the text)
                    part = line.split(", Confidence:")[0].strip()
                    if part:
                        part = part.split(", Recognized text: ")[1].strip()
                    if part:
                        text_only.append(part)
                
                # Join all text in one line separated by space
                clean_text = " ".join(text_only)


        return {"image": str(output_img), "text": clean_text if clean_text else "No text detected"}

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# --------------------- File Upload ---------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    input_path = save_uploaded_image(uploaded_file)

    st.image(uploaded_file, caption="Original Image", use_column_width=True)

    if st.button("🚀 Process Image"):
        with st.spinner("Running StormSight pipeline..."):
            result = run_pipeline(input_path)
            if result:
                st.success("✅ Processing complete!")

                st.subheader("🧼 Derained Image")
                st.image(result["image"], use_column_width=True)

                st.subheader("📝 Extracted Text")
                st.code(result["text"])

                with open(result["image"], "rb") as f:
                    st.download_button(
                        label="💾 Download Derained Image",
                        data=f,
                        file_name="derained_result.png",
                        mime="image/png"
                    )
            else:
                st.error("❌ Failed to process image.")