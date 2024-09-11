import argparse

from api import app

import os

if __name__ == '__main__':
    print(os.getcwd())

    parser = argparse.ArgumentParser()

    parser.add_argument('--host', default='localhost', required=False)
    parser.add_argument('--port', default='8080', required=False, type=int)

    args = parser.parse_args()

    app.run(host=args.host, port=args.port)