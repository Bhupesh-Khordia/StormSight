# 🌧️ StormSight: Robust Scene Text Recognition in Adverse Weather  
*An End-to-End Pipeline for Deraining → Detection → OCR*  

![StormSight Demo](assets/demo.jpeg)  
*Live processing of rainy scene text (Desktop/Jetson compatible)*  

## 🚀 Key Features  
- **Multi-Stage Pipeline**:  
  - 🌧️ **Rain Removal**: Restormer model
  - 🔍 **Text Detection**: CRAFT detector with geometric filtering  
  - 🔠 **Text Recognition**: PARSeq transformer OCR  
  - 🤖 **LLM Enhancement**: Optional Gemini 2.0 Flash post-processing  

## 📂 Project Architecture  
![StormSight Architecture](assets/architecture.png)  

## 🛠️ Installation  

### Prerequisites  
- Python 3.11 (3.6.9 for Jetson)  
- NVIDIA Jetson Nano (Optional for edge deployment)  

### Setup  
```bash
git clone https://github.com/Bhupesh-Khordia/StormSight
cd StormSight

# Create environment (Recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements/base.txt  # or requirements/jetson.txt for edge devices
```

## 💻 Usage Options  

### 1. Command Line  
```bash
# Single image processing
cd src
python pipeline.py

# Live camera processing (Jetson)
cd src_jetson
python text_detection_live.py
```

*Tested by adding synthetic rain to ICDAR-2013 dataset*

## 📝 Academic Context  
**Developed for IC201P Design Practicum at IIT Mandi**  
**Team**:  
- Anshul Mendiratta  
- Bhupesh Yadav  

## 🤝 Contributing  
We welcome contributions! Please see:  
- [Development Guidelines](docs/DEVELOPMENT.md)  
- [Roadmap](docs/ROADMAP.md)  