import os
import webbrowser
import requests
import json
from pathlib import Path

URL = "http://localhost:5000/"


def training():

    response = requests.get(URL + "get_status")
    print(response.json())


if __name__ == '__main__':
    training()
