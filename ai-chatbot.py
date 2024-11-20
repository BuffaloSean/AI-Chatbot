import os
import json
import datetime
import psutil
import requests
import wolframalpha
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

class ToolKit:
    """Manages tool definitions and executions for the chatbot"""
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize external API clients"""
        self.api_keys = api_keys
        self.wolfram_client = wolframalpha.Client(api_keys.get('WOLFRAM_API_KEY'))
    
    def get_weather(self, city: str) -> str:
        """Get weather information for a location"""
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': city,
                'appid': self.api_keys.get('WEATHER_API'),
                'units': 'metric'
            }
            response = requests.get(url, params=params)
            data = response.json()
            print(data)
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return f"Current weather in {city}: {temp}°C, {desc}"
        except Exception as e:
            return f"Error fetching weather data: {str(e)}"

    def get_news(self, topic: Optional[str] = None) -> str:
        """Get latest news articles"""
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'apiKey': self.api_keys.get('NEWS_API_KEY'),
                'language': 'en',
                'q': topic
            }
            response = requests.get(url, params=params)
            articles = response.json()['articles'][:5]
            return "\n".join([f"- {article['title']}" for article in articles])
        except Exception as e:
            return f"Error fetching news: {str(e)}"

    def get_system_metrics(self) -> str:
        """Get current system performance metrics"""
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
            return (f"System Metrics:\nCPU: {metrics['cpu_percent']}%\n"
                   f"Memory: {metrics['memory_percent']}%\n"
                   f"Disk: {metrics['disk_usage']}%")
        except Exception as e:
            return f"Error fetching system metrics: {str(e)}"

    def get_current_time_and_date(self) -> str:
        """Get current time and date"""
        return datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    def ask_wolfram(self, query: str) -> str:
        """Query Wolfram Alpha for information"""
        try:
            res = self.wolfram_client.query(query)
            return next(res.results).text
        except Exception as e:
            return f"Error querying Wolfram Alpha: {str(e)}"

    @classmethod
    def get_tool_definitions(cls) -> List[Dict[str, Any]]:
        """Define available tools and their schemas"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name, e.g. London",
                            }
                        },
                        "required": ["city"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_news",
                    "description": "Get latest news headlines, optionally filtered by topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Optional topic to filter news",
                            }
                        },
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_system_metrics",
                    "description": "Get current system performance metrics",
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time_and_date",
                    "description": "Get the current time and date",
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_wolfram",
                    "description": "Query Wolfram Alpha for factual information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to send to Wolfram Alpha",
                            }
                        },
                        "required": ["query"],
                    },
                }
            }
        ]

class Chatbot:
    """Enhanced chatbot with integrated tools and OpenAI function calling"""
    
    def __init__(self, env_path: str = "api.env"):
        # Load environment variables
        load_dotenv(env_path)
        
        # Collect API keys
        self.api_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'WEATHER_API': os.getenv('WEATHER_API'),
            'NEWS_API_KEY': os.getenv('NEWS_API'),
            'WOLFRAM_API_KEY': os.getenv('WOLFRAMALPHA_API')
        }
        
        # Verify essential API keys
        if not self.api_keys['OPENAI_API_KEY']:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Initialize OpenAI client and toolkit
        self.client = OpenAI(api_key=self.api_keys['OPENAI_API_KEY'])
        self.toolkit = ToolKit(self.api_keys)
        
        # Initialize conversation history
        self.conversation_history = []
        self.system_prompt = """
        You are a helpful assistant with access to various tools. You can:
        1. Check weather for any city
        2. Get latest news updates, optionally filtered by topic
        3. Check system performance metrics
        4. Get current date and time
        5. Query Wolfram Alpha for calculations and facts
        
        Use these tools when appropriate to provide accurate information.
        """
    
    def _format_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Format conversation history and user input for API call"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages
    
    def _execute_tool_call(self, tool_call: Any) -> str:
        """Execute a tool call and return the result"""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Get the function from the ToolKit instance
        function = getattr(self.toolkit, function_name, None)
        if not function:
            return f"Function {function_name} not found. Please try again."
        
        try:
            return function(**function_args)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    def process_message(self, user_input: str) -> str:
        """Process user input and generate response using OpenAI API with function calling"""
        try:
            # Create API request
            completion = self.client.chat.completions.create(
                model='gpt-4o',
                messages=self._format_messages(user_input),
                tool_choice="auto",
                tools=self.toolkit.get_tool_definitions(),
                temperature=0.7
            )
            
            # Extract response components
            message = completion.choices[0].message
            tool_calls = message.tool_calls
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Handle tool calls if present
            if tool_calls:
                responses = []
                for tool_call in tool_calls:
                    result = self._execute_tool_call(tool_call)
                    responses.append(result)
                response = "\n".join(responses)
            else:
                response = message.content or "No response generated."
            
            # Update conversation history with assistant's response
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Trim history if it gets too long
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response
            
        except Exception as e:
            return f"Error processing message: {str(e)}"

def main():
    """Main function to run the chatbot"""
    try:
        # Initialize chatbot
        chatbot = Chatbot()
        
        print("Chatbot initialized. Type 'quit' to exit.")
        print("\nAvailable tools:")
        print("- Weather information for any city")
        print("- Latest news updates")
        print("- System performance metrics")
        print("- Current date and time")
        print("- Wolfram Alpha queries")
        
        # Main conversation loop
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
            
            response = chatbot.process_message(user_input)
            print(f"Assistant: {response}")
            
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")

if __name__ == "__main__":
    main()