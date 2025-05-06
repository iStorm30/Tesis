from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from multiprocessing import Manager
import uvicorn
app = FastAPI()
manager    = Manager()
game_data  = manager.list()
# Aquí guardamos el último comando de movimiento
command_data = manager.dict({"left": False, "right": False, "jump": False})
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
     return game_data

# NUEVO: recibe comando de movimiento desde Python
@app.post("/api/command")
def post_command(cmd: dict):
    command_data.update({
        "left":  cmd.get("left", False),
        "right": cmd.get("right", False),
        "jump":  cmd.get("jump", False),
    })
    return {"status": "ok"}

# NUEVO: Godot consulta el último comando
@app.get("/api/command/latest")
def get_command():
    return dict(command_data)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=4)