from flask import Flask

app = Flask('Soccer Prediction API',
            template_folder='templates',
            static_folder='static',
            # static_url_path='/api/v1/static',
            root_path='api/'
            )

with app.app_context():
    import api.update
    import api.training
    import api.strategy
    import api.e2e
    import api.go_live

