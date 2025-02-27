import firebase_admin
from firebase_admin import db, credentials

class DBController:
    def __init__(self, config):
        self.config = config
        self.setup()
        self.ref = db.reference(config["db_ref"])
        self.clear()

    def setup(self):
        if not firebase_admin._apps:
            cred = credentials.Certificate("src/database/serviceAccountKey.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': self.config["db_url"]
            })
            print("Firebase app initialized.")
    
    def clear(self):
        self.ref.child("balls").delete()

    def update(self, data):
        if isinstance(data, dict) and "balls" in data and data["balls"]:
            self.ref.child("balls").update(data["balls"])

    def cleanup(self):
        firebase_admin.delete_app(firebase_admin.get_app())


