# Deployment Guide — Production Hybrid LLM System

This guide outlines deployment options for the **Hybrid LLM (RAG + LoRA + Semantic Router) System**.

---

## 1. Local Deployment (Bare Metal)

### Prerequisites
* Python 3.10+
* SQLite (built-in)
* **(Optional)** CUDA-compatible GPU with 6GB+ VRAM for 4-bit quantized Mistral-7B.
  * If no GPU is found, the system will automatically fall back to CPU and use `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

### Step 1: Install Dependencies
Activate your virtual environment and install dependencies:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
Key variables inside `.env`:
* `BASE_MODEL_NAME`: Set to `mistralai/Mistral-7B-Instruct-v0.1` for GPU, or `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for CPU testing.
* `DATA_DIR`: Folder for source documents (default: `./data`).
* `PEFT_MODEL_PATH`: Folder where your fine-tuned LoRA adapters are saved (default: `./models/adapters`).

### Step 3: Run the Services
Start the backend API first (Terminal 1):
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the Streamlit UI (Terminal 2):
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
Open **http://localhost:8501** in your browser.

---

## 2. Docker & Docker Compose Deployment (Recommended)

To run the entire system inside isolated containers, use Docker Compose.

```bash
# Build and start both backend and frontend containers
docker-compose up --build -d

# View logs for backend
docker logs -f hybrid_llm_backend

# View logs for frontend
docker logs -f hybrid_llm_frontend
```
* Backend will be accessible at **http://localhost:8000**
* Streamlit UI will be accessible at **http://localhost:8501**

---

## 3. Cloud Platforms Deployments

### A. Railway
1. Fork your project repository to GitHub.
2. In Railway, click **New Project** → **Deploy from GitHub**.
3. Select your repository.
4. Set up two separate services:
   * **Service 1: Backend**
     * Build Command: (Automatic via Dockerfile)
     * Start Command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
     * Environment variables: Set `BASE_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0` (unless you select a GPU plan), `PORT=$PORT`.
   * **Service 2: Frontend**
     * Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
     * Environment variables: `BACKEND_URL` = (the public domain URL of your backend service).

### B. Render
1. Create a **Web Service** for the FastAPI backend.
   * Runtime: `Docker` (Render will build your Dockerfile automatically).
   * Specify environment variables (`BASE_MODEL_NAME`, `DATA_DIR`).
2. Create a **Web Service** or **Static Site** for the Streamlit UI.
   * Runtime: `Docker`
   * Overwrite Docker command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
   * Environment variables: Set `BACKEND_URL` to your backend web service URL.

### C. AWS EC2 (GPU-enabled Instance)
For high-performance inference with Mistral-7B + LoRA adapters:
1. Spin up an EC2 instance with GPU support (e.g. `g4dn.xlarge` with NVIDIA T4).
2. Install Docker and the **NVIDIA Container Toolkit** to allow Docker to access the GPU.
3. Clone your repository and modify `docker-compose.yml` to support GPU mapping:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```
4. Run `docker-compose up --build -d`.
