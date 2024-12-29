#how to initiate flask sample.

from flask import Flask
from waitress import serve
app= Flask('ping')

@app.route('/ping.py', methods = ['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)