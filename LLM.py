from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv
import random

class GeminiLLM(ChatGoogleGenerativeAI):

    def __init__(self , temperature:float = 0.8, model= "gemini-2.5-flash"):
        # Setting API Key
        dotenv.load_dotenv()
        api= os.getenv(self.get_random_key())
        # Initialize LLM
        super().__init__(model=model,temperature=temperature,
                                    api_key= api) #type:ignore

    def get_random_key(self):
        keys = ["GOOGLE_API_KEY", "GEM_PRO", "GEMINI_API_2", "GEMINI_API"]
        return random.choice(keys)

if __name__ == "__main__":
    llm=GeminiLLM(0.0)
    print(llm.invoke(input("Enter your prompt: ")).content)