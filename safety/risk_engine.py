class RiskEngine:

    def __init__(self):
        self.previous_count = 0

    def analyze(self, people_count, density, movement_data):

        risk = "SAFE"
        alerts = []

        if density == "EXTREME":
            risk = "CRITICAL"
            alerts.append("EXTREME CROWD DENSITY")

        elif density == "HIGH":
            risk = "HIGH"
            alerts.append("HIGH CROWD DENSITY")

        fast_movement = 0

        for track_id, speed, direction in movement_data:
            if speed > 25:
                fast_movement += 1

        if fast_movement >= 2:
            if risk == "SAFE":
                risk = "MEDIUM"

            alerts.append("SUDDEN MOVEMENT DETECTED")

        self.previous_count = people_count

        if not alerts:
            alerts.append("NO ACTIVE ALERT")

        return risk, alerts