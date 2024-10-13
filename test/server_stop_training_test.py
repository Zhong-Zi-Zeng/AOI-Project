import os
import webbrowser
import requests
import json
from pathlib import Path

URL = "http://localhost:5000/"


def training():
    json_data = {
        "device_id": "1",
    }
    response = requests.post(URL + "stop_training", json=json_data)
    print(response.json())


if __name__ == '__main__':
    training()
