from fastapi import FastAPI

app = FastAPI(title="Virtual Friend API")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}