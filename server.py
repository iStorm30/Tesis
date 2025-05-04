from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from multiprocessing import Manager
import uvicorn

app = FastAPI()

manager = Manager()
game_data = manager.list()

class Position(BaseModel):
    x: float
    y: float

class GameData(BaseModel):
    lives: int
    souls: int
    position: Position
    end_position: Position
    final: bool
    
# Endpoints para cada dato
#game_data = []

@app.post("/api/update_game_data")
def update_game_data(data: GameData):
    # Agregar un timestamp (opcional) para cada entrada
    entry = {
        "timestamp": datetime.now().isoformat(),
        "lives": data.lives,
        "souls": data.souls,
        "position": {"x": data.position.x, "y": data.position.y},
        "end_position": {"x": data.end_position.x, "y": data.end_position.y},
        "final": data.final

    }
    # Agregar el nuevo registro a la lista
    game_data.append(entry)
    return {"status": "success", "entry": entry}

@app.get("/api/game_data")
def get_game_data():
    return list(game_data)
    # return game_data

@app.get("/api/game_data/latest")
def get_latest_game_data():
    return game_data[-1] if game_data else {"error": "No data available"}

@app.delete("/api/clear_game_data")
def clear_game_data():
    game_data[:] = []  # Limpia la lista compartida
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)