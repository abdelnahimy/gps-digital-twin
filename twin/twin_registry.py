from twin.kalman_filter import VehicleTwin

class TwinRegistry:
    def __init__(self):
        self.twins = {}

    def get_or_create(self, taxi_id: str) -> VehicleTwin:
        """
        Retrieve an existing VehicleTwin for the given taxi_id or create a new one if it doesn't exist.

        Parameters:
        taxi_id (str): Unique identifier for the taxi.

        Returns:
        VehicleTwin: The corresponding VehicleTwin instance.
        """
        if taxi_id not in self.twins:
            self.twins[taxi_id] = VehicleTwin(taxi_id)
        return self.twins[taxi_id]
