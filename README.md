# GPT-SoVITs with Streaming Support

🚧 **Work in Progress** 🚧

This repository is an **extension of the open-source GPT-SoVITs** project originally provided by [RVC-Boss](https://github.com/RVC-Boss).

## What's New?
This version introduces **streaming audio support**, enabling low-latency text-to-speech (TTS) capabilities. Whether you're working on real-time applications or other TTS tasks, this enhancement makes GPT-SoVITs more versatile and powerful.

## How Can I Use It?

## **Method 1: Clone the Repository**

1. **Create a Clone of The Repo:**
```bash
git clone https://github.com/spava002/GPT-SoVITS-Streaming.git
```

2. **Install Requirements:**
```bash
pip install -r requirements.txt
```

---

## **Method 2: Install via Virtual Environment**

1. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**

   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **Linux/Mac**:
     ```bash
     source venv/bin/activate
     ```

3. **Install the Repository:**

   ```bash
   pip install git+https://github.com/spava002/GPT-SoVITS-Streaming.git
   ```

---

## **🔧 Setup Instructions**

1. Open `inference.py`.  
2. Replace the **checkpoint files** with your own.  
   - *If you don’t have checkpoints, refer to the original repository for pretrained models.*  
3. Provide a **reference audio** for inference.  

---

## **▶️ Run Inference**

After setting everything up, simply run the inference to see the results:

```bash
python inference.py
```

---

## Acknowledgments
While developing this extension, inspiration and guidance were also taken from [this repository](https://github.com/JarodMica/GPT-SoVITS-Package), which has explored similar implementations.

## Original Repository
For more details about the original GPT-SoVITs project, please visit the main repository here:  
👉 [GPT-SoVITS by RVC Boss](https://github.com/RVC-Boss/GPT-SoVITS)
