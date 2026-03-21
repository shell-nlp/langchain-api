from functools import lru_cache

from langchain_core.tools import tool
import requests


@tool
@lru_cache
def get_weather(location: str):
    """
    Get the weather for a given location.
    """
    # wttr.in/London
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url)
    data = response.json()
    # 获取当天的天气数据（第一个weather条目）
    current_weather = data["weather"][0]

    # 获取当前时刻的天气条件（使用第一个hourly数据作为当前时刻）
    current_hour = current_weather["hourly"][0]
    conditions = current_hour["weatherDesc"][0]["value"]
    # 提取当前天气信息
    weather_info = {
        "temperature": current_hour["tempC"],  # 温度（摄氏度）
        "conditions": conditions.strip(),  # 天气状况（中文）
        "humidity": current_hour["humidity"],  # 湿度百分比
        "wind_speed": current_hour["windspeedKmph"],  # 风速（公里/小时）
        "feelsLike": current_hour["FeelsLikeC"],  # 体感温度（摄氏度）
    }
    return weather_info


# Usage
if __name__ == "__main__":
    weather = get_weather.invoke({"location": "南京"})
    print(weather)
