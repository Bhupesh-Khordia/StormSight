# 🌧️ StormSight: Reading SCENE Text in Bad Weather Conditions
*[Under Active Development] | IC201P Design Practicum Project*  

## 📌 Overview  
**StormSight** is an ongoing project developing robust scene text recognition for images captured in rainy weather. Implemented as part of **IC201P Design Practicum**, it combines:  
- **Weather Removal**: NERD-Rain for deraining  
- **Text Detection**: CRAFT for localization  
- **Text Recognition**: PARSeq for OCR  

⚠️ *Note: This project is under active development. Results may vary.*  

## 🛠️ Current Implementation Status  
| Component       | Model       | Status          | 
|----------------|------------|----------------|
| Deraining      | NERD-Rain  | ✅ Implemented   |  
| Text Detection | CRAFT      | ✅ Implemented |
| Text Recognition | PARSeq    | ✅ Implemented     | 

## Datasets

[Google Drive](https://drive.google.com/drive/folders/1AHwhVoIo3wW9dvYGbghFSsZcR68g-GcQ?usp=sharing)

## 🚀 Quick Start (Development Preview)  

### Installation  
```bash
# Clone the repository
git clone https://github.com/Bhupesh-Khordia/StormSight 
cd StormSight  

# Create new environment (optional but recommended)

# Windows - 
python -m venv venv

# Linux/MacOS - 
python3 -m venv venv

# Acticate the environment

# Windows -
.\venv\Scripts\activate

# Linux/MacOS -
source venv/bin/activate


# Install dependencies
pip install -r requirements.txt  

# Add images to the input folder
# Place your images in the `data/input/` directory
# Add craft model to models/detection/craft and nerd-rain model to models/deraining/nerd_rain

# Run the pipeline
cd src
python pipeline.py
```

## 🎓 Academic Context
**Developed for IC201P Design Practicum at IIT Mandi.**