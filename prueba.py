import requests
import json

data = {
    "lives": 3,
    "souls": 1,
    "position": {"x": 100, "y": 200},
    "end_position": {"x": 300, "y": 400}
}

response = requests.post("http://127.0.0.1:8000/api/update_game_data", data=json.dumps(data), headers={"Content-Type": "application/json"})

if response.status_code == 200:
    print("Datos enviados correctamente:", response.json())
else:
    print("Error al enviar datos:", response.status_code, response.text)
