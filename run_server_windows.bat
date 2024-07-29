@echo off
powershell -Command "python engine/server.py 2>&1 | tee ./temp/output.log"