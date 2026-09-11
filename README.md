# AI Chatbot (FastAPI + Gemini)

A modern, high-performance conversational AI Chatbot powered by FastAPI, SQLite persistence, and Google Gemini.

---

## 🚀 How to Run in VS Code

### Prerequisites
- **Python 3.10 or higher** installed on your system ([python.org](https://www.python.org/downloads/))
- **Visual Studio Code** installed ([code.visualstudio.com](https://code.visualstudio.com/))
- **Python Extension for VS Code** (search `Python` in VS Code Extensions `Ctrl+Shift+X` or `Cmd+Shift+X`)
- A **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/apikey)

---

### Step 1: Open the Project in VS Code
1. Open Visual Studio Code.
2. Click **File > Open Folder...** and choose this project folder.
3. Open an integrated terminal in VS Code using ``Ctrl + ` `` (or ``Cmd + ` `` on macOS), or from the top menu **Terminal > New Terminal**.

---

### Step 2: Create and Activate a Virtual Environment
In the VS Code terminal, run:

#### On Windows (PowerShell / Command Prompt):
```powershell
# Create virtual environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1
# Or in Command Prompt:
# .\venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

*(You will see `(venv)` appear at the beginning of your terminal prompt).*

---

### Step 3: Install Required Packages
With the virtual environment active, run:
```bash
pip install -r requirements.txt
```

---

### Step 4: Configure Your Gemini API Key
1. In the root of the project, create a file named `.env` (or copy `.env.example` to `.env`).
2. Add your Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(You can get a free API key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)).*

---

### Step 5: Run the Application

You have two easy ways to run the app:

#### Option A: One-Click Launch with VS Code (F5)
1. Press `F5` or click the **Run & Debug** icon on the left sidebar.
2. Select **"Python: Run AI Chatbot (FastAPI)"** and click the green Play button.

#### Option B: Terminal Command
Run directly in the VS Code terminal:
```bash
python main.py
```
*(or `python3 main.py` on Mac/Linux)*

---

### Step 6: Open the Chatbot in Your Browser
Once started, open your web browser and visit:
👉 **[http://localhost:3000](http://localhost:3000)** (or [http://127.0.0.1:3000](http://127.0.0.1:3000))

Enjoy chatting! All conversations and chat history are saved automatically in your local `chatbot.db` SQLite database.
