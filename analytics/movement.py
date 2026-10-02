import math


class MovementAnalyzer:

    def __init__(self):
        self.previous_positions = {}

    def analyze(self, track_id, center_x, center_y):

        direction = "STATIONARY"
        speed = 0.0

        if track_id in self.previous_positions:

            old_x, old_y = self.previous_positions[track_id]

            dx = center_x - old_x
            dy = center_y - old_y

            distance = math.sqrt(
                dx * dx + dy * dy
            )

            speed = distance

            if distance < 3:
                direction = "STATIONARY"

            elif abs(dx) > abs(dy):

                if dx > 0:
                    direction = "RIGHT"
                else:
                    direction = "LEFT"

            else:

                if dy > 0:
                    direction = "DOWN"
                else:
                    direction = "UP"

        self.previous_positions[track_id] = (
            center_x,
            center_y
        )

        return speed, direction