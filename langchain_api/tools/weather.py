from langchain_core.tools import tool


@tool
def get_weather(location: str):
    """
    Get the weather for a given location.
    """
    return {
        "temperature": 20,
        "conditions": "sunny",
        "humidity": 50,
        "wind_speed": 10,
        "feelsLike": 25,
    }
