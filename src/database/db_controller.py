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
        self.ref.child("balls").set({})

    def update(self, data):
        if isinstance(data, dict) and "balls" in data and data["balls"]:
            # Get previous state from Firebase
            previous_balls = self.ref.child("balls").get() or {}
            # Track currently detected balls
            detected_balls = {}

            # Update existing balls
            for ball_type, ball_data in data["balls"].items():
                detected_balls[ball_type] = []
                for ball_id, ball_values in enumerate(ball_data):
                    detected_balls[ball_type].append(str(ball_id)) 
                    self.ref.child(f"balls/{ball_type}/{ball_id}").update(ball_values)

            # Remove balls that are no longer present
            for ball_type, prev_ball_data in previous_balls.items():
                if ball_type not in detected_balls:
                    self.ref.child(f"balls/{ball_type}").delete()  # Remove entire type if missing
                    continue
                
                if isinstance(prev_ball_data, list):
                    for ball_id in range(len(prev_ball_data)): 
                        if str(ball_id) not in detected_balls[ball_type]:  # If ball is missing, remove it
                            self.ref.child(f"balls/{ball_type}/{ball_id}").delete()


    def cleanup(self):
        firebase_admin.delete_app(firebase_admin.get_app())


