from pydantic import BaseModel

class DeliveryInput(BaseModel):
    road_traffic_density: int
    delivery_person_rating: float
    vehicle_condition: int
    type_of_order: int
    type_of_vehicle: int
