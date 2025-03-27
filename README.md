# L4D2-LLMChat

**Enhance your Left 4 Dead 2 experience by enabling survivor bots to chat intelligently using various Large Language Models (LLMs)!**

This project is an enhanced fork of **smilz0's original Dot4GPT** ([https://github.com/smilz0/Dot4GPT](https://github.com/smilz0/Dot4GPT)). It expands upon the original concept by adding support for multiple LLM providers and managing all survivor bots within a single application instance.

## Features

* **Multi-LLM Support:** Connect to various AI services:
  * OpenAI (GPT models)
  * Ollama (Run models like Llama3, Mistral, etc., locally)
  * Google Gemini
  * Mistral AI API
  * Groq (Fast Llama3, Mixtral inference)
* **Single Instance, Multi-Bot:** Run just **one** instance of the application to handle chat for all 8 survivors.
* **Flexible Configuration:** Easily switch between LLM providers and configure API keys/models via a `settings.json` file.
* **In-Character Responses:** Uses system prompts to instruct the LLM to respond according to the survivor's personality.
* **Conversation Context:** Maintains separate conversation histories for each bot.
* **Idle Timeout:** Automatically resets a bot's conversation context after a period of inactivity.
* **Real-time(ish) Interaction:** Monitors input files generated by the companion L4D2 addon and writes responses for the addon to read.

## Side Note

Since this is a multi-bot instance, certain adjustments in chat formatting can significantly improve response quality. Based on experiments, it's best to avoid commas and formalities when chatting with the bots. The system is optimized for quick, dynamic exchanges, and structuring your input correctly enhances bot responsiveness.

For optimal results:
- **Start your message with the bot's name,** followed directly by your chat. Example:
  - `coach how are you doing?`
  - `nick is coach your best buddy?`
- **Keep interactions concise and direct** for a seamless experience.
- **You can spam chat**, but it's best to avoid multiple instances of chat at the same time, especially if you're running local LLMs. This prevents excessive resource usage, avoids running multiple requests at once, and helps **reduce hallucinations** in responses.

This approach ensures smooth bot-switching, maintains clarity in conversations, and optimizes system performance.

## Requirements

1. **Left 4 Dead 2**
2. **Left 4 GPT Addon** (Available on Steam Workshop, comes with Left 4 Lib Addon)
3. **Left 4 Lib Addon** (If not available on your addon)
4. **.NET Runtime** (Version 6.0 or later) - [Download Here](https://dotnet.microsoft.com/en-us/download/dotnet/6.0)
5. **API Keys / Local LLMs:**
   * OpenAI, Gemini, Mistral, Groq (API keys required)
   * Ollama (Local server & models required)

## Installation & Setup

1. **Download & Install Requirements**
2. **Get Application Files:** Download the latest release from [Releases](https://github.com/YOUR_USERNAME/L4D2-LLMChat/releases)
3. **Extract/Place Files:** Put files in a dedicated folder (e.g., `C:\L4D2-LLMChat`)
4. **First Run:** Run `L4D2-LLMChat.exe` to generate `settings.json`
5. **Configure `settings.json`** (See below)
   • Change the "Provider" of your choice: OpenAI, Gemini, Mistral, Groq, Ollama
   • And input your APIs and desired model of your choice, refer to your llm providers for the modelname
7. **Run the Application**: Launch `L4D2-LLMChat.exe`

## Configuration (`settings.json`)

```json
{
  "Provider": "Ollama",
  "IOPath": "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Left 4 Dead 2\\left4dead2\\ems\\left4gpt",
  "ApiKey": "sk-...",
  "OpenAiModel": "gpt-3.5-turbo",
  "OllamaBaseUrl": "http://localhost:11434",
  "OllamaModel": "llama3",
  "GeminiApiKey": "AIzaSy...",
  "GeminiModel": "gemini-1.5-flash-latest",
  "MistralApiKey": "...",
  "MistralModel": "mistral-large-latest",
  "GroqApiKey": "gsk_...",
  "GroqModel": "llama3-8b-8192",
  "MaxTokens": 60,
  "MaxContext": 5,
  "ResetIdleSeconds": 120,
  "APIErrors": true
}
```

## Running the Application

1. Configure `settings.json` correctly.
2. Ensure local services (like Ollama) are running if selected.
3. Run `L4D2-LLMChat.exe` (Console window should appear).
4. Keep the application running while playing L4D2.
5. Chat messages in-game will now trigger bot responses.
Note: If at first it will not respond, just try chatting it again or you could check the cmd terminal if it's functioning properly.

## How It Works

1. Player types a message in L4D2.
2. The VScript addon writes the message to an input file.
3. The application detects the change, processes it, and sends it to the LLM.
4. The response is written to an output file.
5. The VScript addon reads the response and makes the bot say it.

## Acknowledgements

* Based on **Dot4GPT** by smilz0 ([GitHub](https://github.com/smilz0/Dot4GPT))
* Thanks to developers of the various LLM APIs and .NET libraries used.

