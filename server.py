from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from multiprocessing import Manager
from typing import Optional
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

# Mantiene el último comando enviado por la IA
command_data = {
    "left": False,
    "right": False,
    "jump": False,
    "attack" : False
}


@app.post("/api/update_game_data")
def update_game_data(data: GameData):
    # Agregar un timestamp (opcional) para cada entrada
    entry = {
        "timestamp": datetime.now().isoformat(),
        "lives": data.lives,
        "souls": data.souls,
        "position": {"x": data.position.x, "y": data.position.y},
        "end_position": {"x": data.end_position.x, "y": data.end_position.y},
        "final": data.final,

    }
    # Agregar el nuevo registro a la lista
    game_data.append(entry)
    return {"status": "success", "entry": entry}

@app.get("/api/game_data")
def get_game_data():
     return game_data

@app.post("/api/command")
def post_command(cmd: dict):
    """
    Recibe un JSON con {'left': bool, 'right': bool, 'jump': bool}
    y actualiza command_data.
    """
    command_data.update({
        "left":  cmd.get("left", False),
        "right": cmd.get("right", False),
        "jump":  cmd.get("jump", False),
        "attack": cmd.get("attack",False)
    })
    return {"status": "ok"}

@app.get("/api/command/latest")
def get_command():
    """
    Devuelve el último comando para que Godot lo consulte.
    """
    return command_data

@app.post("/api/level_event")
def level_event(evt: dict):
    # evt = {"timer":0.0, "reset":True}
    entry = {"timestamp": datetime.now().isoformat(), **evt}
    game_data.append(entry)
    
    return {"status":"ok"}