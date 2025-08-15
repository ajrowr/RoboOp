
import requests

from robo import *
from robo.tools import Tool


class ToolsTesterTravelPlanner(Bot):
    sysprompt_text = """You are a travel planning assistant. When users ask about travel between cities, you must:
    1. First get weather information for the departure city
    2. Then get weather information for the destination city  
    3. Finally, calculate the route distance and travel time between the two cities
    
    You MUST call these tools in this exact sequence for every travel query. Always provide a comprehensive comparison of both cities' weather before giving route information."""
    
    class GetCityWeather(Tool):
        description = "Get current weather conditions for a specific city"
        parameter_descriptions = {
            'city_name': "Name of the city to get weather for",
        }
        def __call__(self, city_name:str):
            import random
            weather_conditions = ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy']
            return {
                'city_name': city_name,
                'temperature_celsius': round(random.uniform(-10, 35), 1),
                'condition': random.choice(weather_conditions),
                'humidity': random.randint(30, 90),
                'wind_speed_kmh': round(random.uniform(0, 25), 1)
            }
        
    class CalculateRoute(Tool):
        description = "Calculate distance and travel time between two cities. Only call this after getting weather for both cities."
        parameter_descriptions = {
            'origin_city': "Starting city name",
            'destination_city': "Destination city name"
        }
        def __call__(self, origin_city:str, destination_city:str):
            import random
            # Simulate route calculation
            distance_km = random.randint(50, 1500)
            drive_time_hours = round(distance_km / random.randint(60, 120), 1)
            flight_time_hours = round(distance_km / random.randint(400, 800), 1)
        
            return {
                'origin': origin_city,
                'destination': destination_city,
                'distance_km': distance_km,
                'estimated_drive_time_hours': drive_time_hours,
                'estimated_flight_time_hours': flight_time_hours,
                'recommended_transport': 'flight' if distance_km > 500 else 'car'
            }
    
    tools = [GetCityWeather, CalculateRoute]


class GuidedNavigationTester(Bot):
    """Guided navigation is a feature designed for website chatbots to allow them to
    "guide" the user around a site by sending special navigation messages that are caught
    by client-side code.
    """
    sysprompt_text = """You are a testing assistant. The user may ask you to navigate them to places, in which case you use the guided navigation tool to do this. Do not include any text before or after your tool call. When the navigation event completes you will receive an acknowledgement in the form "@@@@RECONNECT" and you may continue the conversation. Other than this, proceed with the conversation as normal."""
    
    def get_tools_schema(self):
        return [
            {
                'name': 'guided_navigate',
                'description': "Navigates a user to a specific link on the site.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": "The destination link on the site to navigate the user to."
                        }
                    },
                    "required": ["destination"]
                }
            }
        ]
    
    def tools_guided_navigate(self, destination=None):
        return {
            "message": f'@@@@NAVIGATE {destination}',
            "target": "client"
        }
    
    def preprocess_response(self, message, conversation):
        """If we got a RECONNECT then we need to match the tool use ID up"""
        if message.startswith('@@@@RECONNECT'):
            tu_id = conversation._get_last_tool_use_id()
            # conversation.messages.append()
            # return ('@@@@RECONNECT', True)
            return conversation._make_tool_result_message({'id': tu_id}, "@@@@RECONNECT")
    

class FetchAndAnalyseBot(Bot):
    sysprompt_text = """Your task is to retrieve and analyse the contents of either a provided URL."""
    
    class GetURL(Tool):
        description = 'Fetch the raw HTML from a given URL'
        parameter_descriptions = {
            'url': 'The URL to fetch',
        }
        def __call__(self, url:str):
            print(f"\n[[Fetching URL: {url}]]")
            pagetext = requests.get(url).text
            return pagetext
    
    tools = [GetURL]

