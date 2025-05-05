from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import uvicorn

app = FastAPI()


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
game_data = []

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

