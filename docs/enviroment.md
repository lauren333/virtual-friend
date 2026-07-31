# VirtualFriend — Development Setup

## Frontend: React + TypeScript + Vite 
    #npm run dev
    -Port 5173 
    -Vite starts development server. React runs in browser, vite serves frontend code to it.  

## Backend: Python + FastAPI 
    #source .venv/bin/activate 
    #fastapi dev app/main.py

    -Using a virtual enviromet to keep Python dependencies isolated from other projects on the machine.
    -backend should start up on http://127.0.0.1:8000 
    -Visit http://127.0.0.1:8000/api/health to check for the expected response or via Swagger UI. 


## ML - Open CV, PyTorch 
    #cd ~/Documents/virtualfriend/ml
    #source .venv/bin/activate