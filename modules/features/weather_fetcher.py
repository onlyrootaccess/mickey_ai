# OpenWeatherMap API
"""
Mickey AI - Weather Fetcher
Real-time weather data from OpenWeatherMap API with location intelligence
"""

import logging
import requests
import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import geocoder
from geopy.geocoders import Nominatim
import reverse_geocoder as rg

class WeatherCondition(Enum):
    CLEAR = "clear"
    CLOUDS = "clouds"
    RAIN = "rain"
    DRIZZLE = "drizzle"
    THUNDERSTORM = "thunderstorm"
    SNOW = "snow"
    MIST = "mist"
    SMOKE = "smoke"
    HAZE = "haze"
    DUST = "dust"
    FOG = "fog"
    SAND = "sand"
    ASH = "ash"
    SQUALL = "squall"
    TORNADO = "tornado"

class TemperatureUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"

class WeatherFetcher:
    def __init__(self, api_key: str = None, cache_duration: int = 600):
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.geocoding_url = "http://api.openweathermap.org/geo/1.0"
        
        # Cache for weather data
        self.weather_cache = {}
        self.cache_duration = cache_duration  # 10 minutes default
        
        # Location services
        self.geolocator = Nominatim(user_agent="mickey_ai_weather")
        self.current_location = None
        
        # Weather icons mapping
        self.weather_icons = {
            "01d": "☀️",  # clear sky day
            "01n": "🌙",  # clear sky night
            "02d": "⛅",  # few clouds day
            "02n": "☁️",  # few clouds night
            "03d": "☁️",  # scattered clouds
            "03n": "☁️",  # scattered clouds
            "04d": "☁️",  # broken clouds
            "04n": "☁️",  # broken clouds
            "09d": "🌧️",  # shower rain
            "09n": "🌧️",  # shower rain
            "10d": "🌦️",  # rain day
            "10n": "🌧️",  # rain night
            "11d": "⛈️",  # thunderstorm day
            "11n": "⛈️",  # thunderstorm night
            "13d": "❄️",  # snow day
            "13n": "❄️",  # snow night
            "50d": "🌫️",  # mist day
            "50n": "🌫️",  # mist night
        }
        
        # Mickey's weather personalities
        self.weather_messages = {
            "sunny": [
                "Perfect weather! Mickey suggests going outside! ☀️",
                "Sunshine alert! Great day for an adventure! 🌞",
                "Beautiful sunny day! Mickey's sunglasses are on! 😎"
            ],
            "cloudy": [
                "Cloudy but cozy! Perfect for indoor activities! ☁️",
                "Partly cloudy! Mickey says it's still a good day! ⛅",
                "Clouds in the sky, but smiles all around! 😊"
            ],
            "rainy": [
                "Rainy day! Perfect for reading with Mickey! 📚",
                "Don't forget your umbrella! Mickey's staying dry! ☔",
                "Rainy weather! Time for some indoor fun! 🎮"
            ],
            "stormy": [
                "Stormy weather! Mickey's staying safe inside! ⚡",
                "Thunder and lightning! Better stay indoors! 🌩️",
                "Storm alert! Mickey recommends cozy activities! 🏠"
            ],
            "snowy": [
                "Snow day! Mickey wants to build a snowman! ⛄",
                "Brrr! Cold weather! Time for hot chocolate! ☕",
                "Snowfall! Mickey's making snow angels! ❄️"
            ],
            "hot": [
                "Hot day! Mickey's staying cool with ice cream! 🍦",
                "Heat wave! Stay hydrated, friends! 💧",
                "Scorching weather! Perfect for the beach! 🏖️"
            ],
            "cold": [
                "Chilly weather! Mickey's bundling up! 🧣",
                "Cold day ahead! Hot drinks recommended! ☕",
                "Brrr! Mickey says wear something warm! 🧥"
            ]
        }
        
        self.logger.info("🌤️ Weather Fetcher initialized - Ready for forecasts!")

    def set_api_key(self, api_key: str):
        """Set OpenWeatherMap API key"""
        self.api_key = api_key
        self.logger.info("Weather API key configured")

    def get_current_weather(self, location: str = None, unit: TemperatureUnit = TemperatureUnit.CELSIUS) -> Dict[str, Any]:
        """
        Get current weather for a location
        
        Args:
            location: City name or coordinates (lat,lon)
            unit: Temperature unit
            
        Returns:
            Dictionary with current weather data
        """
        try:
            if not self.api_key:
                return self._create_error_response("Weather API key not configured")

            # Get coordinates for location
            if location:
                coords = self._get_coordinates(location)
            else:
                coords = self._get_current_location()
                
            if not coords:
                return self._create_error_response("Could not determine location")

            # Check cache
            cache_key = f"current_{coords['lat']}_{coords['lon']}_{unit.value}"
            cached_data = self._get_cached_weather(cache_key)
            if cached_data:
                self.logger.info("Using cached weather data")
                return cached_data

            # Build API URL
            url = f"{self.base_url}/weather"
            params = {
                'lat': coords['lat'],
                'lon': coords['lon'],
                'appid': self.api_key,
                'units': self._get_unit_param(unit)
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            weather_data = self._parse_current_weather(data, unit, coords)
            
            # Cache the result
            self._cache_weather(cache_key, weather_data)
            
            self.logger.info(f"Weather fetched for {coords.get('name', 'unknown location')}")
            return weather_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Weather API request failed: {str(e)}")
            return self._create_error_response(f"Weather service unavailable: {str(e)}")
        except Exception as e:
            self.logger.error(f"Weather fetch failed: {str(e)}")
            return self._create_error_response(f"Weather fetch failed: {str(e)}")

    def get_weather_forecast(self, location: str = None, days: int = 5, 
                           unit: TemperatureUnit = TemperatureUnit.CELSIUS) -> Dict[str, Any]:
        """
        Get weather forecast for a location
        
        Args:
            location: City name or coordinates
            days: Number of days to forecast (1-5)
            unit: Temperature unit
            
        Returns:
            Dictionary with weather forecast
        """
        try:
            if not self.api_key:
                return self._create_error_response("Weather API key not configured")

            # Validate days parameter
            days = max(1, min(5, days))

            # Get coordinates
            if location:
                coords = self._get_coordinates(location)
            else:
                coords = self._get_current_location()
                
            if not coords:
                return self._create_error_response("Could not determine location")

            # Check cache
            cache_key = f"forecast_{coords['lat']}_{coords['lon']}_{days}_{unit.value}"
            cached_data = self._get_cached_weather(cache_key)
            if cached_data:
                self.logger.info("Using cached forecast data")
                return cached_data

            # Build API URL
            url = f"{self.base_url}/forecast"
            params = {
                'lat': coords['lat'],
                'lon': coords['lon'],
                'appid': self.api_key,
                'units': self._get_unit_param(unit),
                'cnt': days * 8  # 8 forecasts per day
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecast_data = self._parse_forecast(data, days, unit, coords)
            
            # Cache the result
            self._cache_weather(cache_key, forecast_data)
            
            self.logger.info(f"Forecast fetched for {coords.get('name', 'unknown location')} ({days} days)")
            return forecast_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Forecast API request failed: {str(e)}")
            return self._create_error_response(f"Weather service unavailable: {str(e)}")
        except Exception as e:
            self.logger.error(f"Forecast fetch failed: {str(e)}")
            return self._create_error_response(f"Forecast fetch failed: {str(e)}")

    def _get_coordinates(self, location: str) -> Optional[Dict[str, Any]]:
        """Get coordinates for a location string"""
        try:
            # Try OpenWeatherMap geocoding first
            if self.api_key:
                url = f"{self.geocoding_url}/direct"
                params = {
                    'q': location,
                    'limit': 1,
                    'appid': self.api_key
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        return {
                            'lat': data[0]['lat'],
                            'lon': data[0]['lon'],
                            'name': data[0]['name'],
                            'country': data[0]['country']
                        }
            
            # Fallback to geopy
            location_data = self.geolocator.geocode(location)
            if location_data:
                return {
                    'lat': location_data.latitude,
                    'lon': location_data.longitude,
                    'name': location_data.address.split(',')[0],
                    'country': location_data.address.split(',')[-1].strip()
                }
            
            self.logger.warning(f"Could not find coordinates for: {location}")
            return None
            
        except Exception as e:
            self.logger.error(f"Geocoding failed for {location}: {str(e)}")
            return None

    def _get_current_location(self) -> Optional[Dict[str, Any]]:
        """Get current location using IP geolocation"""
        try:
            if self.current_location:
                return self.current_location
            
            # Get location by IP
            g = geocoder.ip('me')
            if g.ok:
                self.current_location = {
                    'lat': g.latlng[0],
                    'lon': g.latlng[1],
                    'name': g.city,
                    'country': g.country
                }
                return self.current_location
            
            # Fallback to reverse geocoding with default coordinates
            self.logger.warning("Using default location (New York)")
            return {
                'lat': 40.7128,
                'lon': -74.0060,
                'name': 'New York',
                'country': 'US'
            }
            
        except Exception as e:
            self.logger.error(f"Current location detection failed: {str(e)}")
            return {
                'lat': 40.7128,
                'lon': -74.0060,
                'name': 'New York',
                'country': 'US'
            }

    def _parse_current_weather(self, data: Dict, unit: TemperatureUnit, coords: Dict) -> Dict[str, Any]:
        """Parse current weather data from API response"""
        main = data['main']
        weather = data['weather'][0]
        wind = data.get('wind', {})
        
        # Determine weather category
        weather_category = self._get_weather_category(weather['main'], main['temp'], unit)
        
        return {
            'success': True,
            'location': {
                'name': coords.get('name', data.get('name', 'Unknown')),
                'country': coords.get('country', 'Unknown'),
                'coordinates': {
                    'latitude': coords['lat'],
                    'longitude': coords['lon']
                }
            },
            'temperature': {
                'current': round(main['temp'], 1),
                'feels_like': round(main['feels_like'], 1),
                'min': round(main['temp_min'], 1),
                'max': round(main['temp_max'], 1),
                'unit': unit.value
            },
            'conditions': {
                'main': weather['main'],
                'description': weather['description'].title(),
                'icon': self.weather_icons.get(weather['icon'], '🌈'),
                'category': weather_category
            },
            'additional': {
                'humidity': main['humidity'],
                'pressure': main['pressure'],
                'wind_speed': wind.get('speed', 0),
                'wind_direction': wind.get('deg', 0),
                'cloudiness': data['clouds']['all'],
                'visibility': data.get('visibility', 0),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
            },
            'timestamp': datetime.now().isoformat(),
            'mickey_response': self._get_weather_message(weather_category, main['temp'], unit)
        }

    def _parse_forecast(self, data: Dict, days: int, unit: TemperatureUnit, coords: Dict) -> Dict[str, Any]:
        """Parse forecast data from API response"""
        daily_forecasts = []
        
        # Group forecasts by day
        daily_data = {}
        for forecast in data['list']:
            date = forecast['dt_txt'].split(' ')[0]  # Get date part
            if date not in daily_data:
                daily_data[date] = []
            daily_data[date].append(forecast)
        
        # Process each day
        for date, forecasts in list(daily_data.items())[:days]:
            temps = [f['main']['temp'] for f in forecasts]
            weather_counts = {}
            
            for forecast in forecasts:
                weather_main = forecast['weather'][0]['main']
                weather_counts[weather_main] = weather_counts.get(weather_main, 0) + 1
            
            # Determine dominant weather condition
            dominant_weather = max(weather_counts.items(), key=lambda x: x[1])[0]
            
            daily_forecasts.append({
                'date': date,
                'temperature': {
                    'min': round(min(temps), 1),
                    'max': round(max(temps), 1),
                    'unit': unit.value
                },
                'conditions': {
                    'main': dominant_weather,
                    'description': dominant_weather.title(),
                    'icon': self.weather_icons.get(forecasts[0]['weather'][0]['icon'], '🌈'),
                    'category': self._get_weather_category(dominant_weather, sum(temps)/len(temps), unit)
                },
                'humidity': round(sum(f['main']['humidity'] for f in forecasts) / len(forecasts)),
                'wind_speed': round(sum(f['wind']['speed'] for f in forecasts) / len(forecasts), 1)
            })
        
        return {
            'success': True,
            'location': {
                'name': coords.get('name', data['city']['name']),
                'country': coords.get('country', data['city']['country']),
                'coordinates': {
                    'latitude': coords['lat'],
                    'longitude': coords['lon']
                }
            },
            'forecast_days': days,
            'daily_forecasts': daily_forecasts,
            'timestamp': datetime.now().isoformat(),
            'mickey_response': f"Mickey's got your {days}-day forecast! 🌈"
        }

    def _get_weather_category(self, condition: str, temperature: float, unit: TemperatureUnit) -> str:
        """Categorize weather for Mickey's responses"""
        # Convert to Celsius for categorization if needed
        if unit == TemperatureUnit.FAHRENHEIT:
            temp_c = (temperature - 32) * 5/9
        elif unit == TemperatureUnit.KELVIN:
            temp_c = temperature - 273.15
        else:
            temp_c = temperature
        
        # Temperature-based categories
        if temp_c > 30:
            temp_category = "hot"
        elif temp_c > 20:
            temp_category = "warm"
        elif temp_c > 10:
            temp_category = "mild"
        elif temp_c > 0:
            temp_category = "cool"
        else:
            temp_category = "cold"
        
        # Condition-based categories
        condition_lower = condition.lower()
        if condition_lower in ['clear']:
            return "sunny"
        elif condition_lower in ['clouds', 'mist', 'haze', 'fog']:
            return "cloudy"
        elif condition_lower in ['rain', 'drizzle']:
            return "rainy"
        elif condition_lower in ['thunderstorm']:
            return "stormy"
        elif condition_lower in ['snow']:
            return "snowy"
        else:
            return temp_category

    def _get_weather_message(self, category: str, temperature: float, unit: TemperatureUnit) -> str:
        """Get Mickey's personalized weather message"""
        import random
        
        # Get base message
        messages = self.weather_messages.get(category, ["Weather update from Mickey! 🌤️"])
        base_message = random.choice(messages)
        
        # Add temperature context
        temp_str = f"{temperature:.1f}°"
        if unit == TemperatureUnit.FAHRENHEIT:
            temp_str += "F"
        elif unit == TemperatureUnit.CELSIUS:
            temp_str += "C"
        else:
            temp_str += "K"
        
        return f"{base_message} Current temperature: {temp_str}"

    def _get_unit_param(self, unit: TemperatureUnit) -> str:
        """Get API unit parameter"""
        if unit == TemperatureUnit.FAHRENHEIT:
            return "imperial"
        elif unit == TemperatureUnit.CELSIUS:
            return "metric"
        else:
            return "standard"  # Kelvin

    def _get_cached_weather(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached weather data if valid"""
        if cache_key in self.weather_cache:
            cached_time, data = self.weather_cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return data
            else:
                # Remove expired cache
                del self.weather_cache[cache_key]
        return None

    def _cache_weather(self, cache_key: str, data: Dict[str, Any]):
        """Cache weather data"""
        self.weather_cache[cache_key] = (time.time(), data)
        
        # Clean up old cache entries
        current_time = time.time()
        self.weather_cache = {
            k: v for k, v in self.weather_cache.items() 
            if current_time - v[0] < self.cache_duration * 2
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'mickey_response': random.choice([
                "Oops! Mickey can't check the weather right now! 🌧️",
                "Weather service is taking a break! Try again soon! 🌞",
                "Mickey's weather magic isn't working! 😅"
            ])
        }

    def get_weather_alerts(self, location: str = None) -> Dict[str, Any]:
        """Get weather alerts for a location (placeholder for future implementation)"""
        # Note: This would require a premium OpenWeatherMap subscription
        return {
            'success': False,
            'message': 'Weather alerts require premium API subscription',
            'mickey_response': "Mickey's weather alerts are coming soon! 🚨"
        }

    def set_cache_duration(self, duration: int):
        """Set cache duration in seconds"""
        self.cache_duration = duration
        self.logger.info(f"Weather cache duration set to {duration} seconds")

    def clear_cache(self):
        """Clear weather cache"""
        self.weather_cache.clear()
        self.logger.info("Weather cache cleared")

    def get_weather_stats(self) -> Dict[str, Any]:
        """Get weather fetcher statistics"""
        return {
            'cache_size': len(self.weather_cache),
            'cache_duration_seconds': self.cache_duration,
            'api_configured': self.api_key is not None,
            'current_location': self.current_location,
            'supported_units': [unit.value for unit in TemperatureUnit]
        }

# Test function
def test_weather_fetcher():
    """Test the weather fetcher (requires API key)"""
    # Note: Replace with actual API key for testing
    weather = WeatherFetcher(api_key="your_api_key_here")
    
    print("Testing Weather Fetcher...")
    
    # Test current weather
    current = weather.get_current_weather("London", TemperatureUnit.CELSIUS)
    print("Current Weather:", current.get('mickey_response', 'Error'))
    
    if current['success']:
        print(f"Temperature: {current['temperature']['current']}°C")
        print(f"Conditions: {current['conditions']['description']}")
    
    # Test forecast
    forecast = weather.get_weather_forecast("New York", days=3, unit=TemperatureUnit.FAHRENHEIT)
    print("Forecast:", forecast.get('mickey_response', 'Error'))
    
    if forecast['success']:
        for day in forecast['daily_forecasts']:
            print(f"{day['date']}: {day['temperature']['min']}°F - {day['temperature']['max']}°F")
    
    # Test stats
    stats = weather.get_weather_stats()
    print("Weather Stats:", stats)

if __name__ == "__main__":
    test_weather_fetcher()