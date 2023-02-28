# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("hello world")
from flask import Flask

app = Flask(__name__)
@app.route('/')
def hello_world():
        return "hello world,web"
    
app.run()