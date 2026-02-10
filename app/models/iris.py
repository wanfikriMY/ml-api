from pydantic import BaseModel


class Iris(BaseModel):
    data: list[list[float]]
