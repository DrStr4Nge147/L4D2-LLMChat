using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

// Add any using statements needed by your specific service implementations
// For example, if using default OpenAI model name:
using OpenAI.GPT3.ObjectModels;

// Assumes BotState is defined in BotState.cs
// Assumes Settings is defined in Settings.cs (or below if kept here)
// Assumes ILLMService is defined in ILLMService.cs
// Assumes GenericChatMessage/ChatRole are defined in Models.cs

public class Program
{
    // --- Configuration ---
    const string SettingsFileName = @"settings.json";
    // Define all the bots this instance should manage
    static readonly string[] SurvivorNames = { "bill", "francis", "louis", "zoey", "coach", "ellis", "nick", "rochelle" };

    // --- State Management ---
    // Using ConcurrentDictionary for thread safety with FileSystemWatcher callbacks
    // Assumes BotState class (with IsProcessing flag) is defined elsewhere
    static ConcurrentDictionary<string, BotState> botStates = new ConcurrentDictionary<string, BotState>();

    // --- Shared Services ---
    static Settings? appSettings; // Hold loaded settings
    static ILLMService? llmService; // Hold initialized service
    static string? ioPath; // Hold IO Path

    // --- Debouncing State ---
    // Dictionary to track the last time an event was *scheduled* for processing
    private static readonly ConcurrentDictionary<string, DateTime> _lastEventScheduledTime = new ConcurrentDictionary<string, DateTime>();
    private const int DebounceMilliseconds = 250; // Wait 250ms for stability before processing

    // --- Processing Lock Object ---
    private static readonly object _processingLock = new object();


    // --- Main Entry Point ---
    public static async Task<int> Main(string[] args) // Args are not used
    {
        Console.WriteLine("--- Dot4GPT Multi-Bot (Watcher + Debounce + Lock Mode) Initializing ---");

        // --- Load Settings ---
        appSettings = LoadSettings(); // Store settings globally
        if (appSettings == null)
        {
            Console.WriteLine("Exiting due to settings load failure.");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
            return 1;
        }
        ioPath = AddTrailingSeparator(appSettings.IOPath); // Store IO Path globally

        // --- Initialize Shared LLM Service ---
        try
        {
            Console.WriteLine($"Attempting to initialize provider: {appSettings.Provider}");
            llmService = CreateLlmService(appSettings); // Store service globally
            Console.WriteLine("LLM Service Initialized Successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"!! FATAL: Error initializing LLM service for provider '{appSettings.Provider}': {ex.Message}");
            Console.WriteLine($"Details: {ex.ToString()}");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
            return 1;
        }

        // --- Initialize State and Watchers for Each Bot ---
        Console.WriteLine($"Using IO Path: {ioPath}");
        List<FileSystemWatcher> watchers = new List<FileSystemWatcher>(); // Keep track of watchers

        foreach (string botName in SurvivorNames)
        {
            Console.WriteLine($"Initializing state for: {Capitalize(botName)}");
            var newState = new BotState(); // Assumes BotState definition exists
            newState.IsProcessing = false; // Explicitly initialize flag
            string systemPrompt = GenerateSystemPrompt(botName, appSettings.MaxTokens);
            newState.Messages.Add(GenericChatMessage.FromSystem(systemPrompt)); // Assumes GenericChatMessage exists
            newState.LastInteraction = DateTime.Now;

            if (!botStates.TryAdd(botName, newState))
            {
                Console.WriteLine($"!! Warning: Failed to add initial state for {botName}. Skipping watcher setup.");
                continue;
            }
            Console.WriteLine($" -> System Prompt: {systemPrompt}");

            // --- Setup FileSystemWatcher for this bot's input file ---
            try
            {
                if (!Directory.Exists(ioPath))
                {
                    Console.WriteLine($"!! FATAL: IO Path directory does not exist: {ioPath}");
                    Console.WriteLine("Press any key to exit...");
                    Console.ReadKey();
                    return 1;
                }

                var watcher = new FileSystemWatcher(ioPath)
                {
                    Filter = $"{botName}_in.txt",
                    NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.CreationTime,
                    EnableRaisingEvents = true,
                    IncludeSubdirectories = false
                };

                string currentBotName = botName;
                watcher.Changed += (sender, e) => HandleFileEventDebounced(e, currentBotName);
                watcher.Created += (sender, e) => HandleFileEventDebounced(e, currentBotName);

                watchers.Add(watcher);
                Console.WriteLine($"   -> Watching file: {watcher.Filter}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"!! Error setting up FileSystemWatcher for {botName}: {ex.Message}");
            }
        }
        Console.WriteLine("--- All Bots Initialized and Watchers Started ---");
        Console.WriteLine("--- Waiting for input file changes... (Press Ctrl+C to exit) ---");

        // --- Keep Main Thread Alive & Check Idle ---
        while (true)
        {
            if (appSettings != null)
            {
                foreach (string botName in SurvivorNames)
                {
                    CheckBotIdleTimeout(botName, appSettings);
                }
            }
            await Task.Delay(5000); // Check idle status every 5 seconds
        }
        // Unreachable code below unless loop is broken
        // watchers.ForEach(w => { w.EnableRaisingEvents = false; w.Dispose(); });
        // return 0; // Needed if Main declared as Task<int>
    }

    // --- Debounced Event Handler ---
    private static void HandleFileEventDebounced(FileSystemEventArgs e, string botName)
    {
        // Basic validation - check filename carefully
        if (!Path.GetFileName(e.FullPath).Equals($"{botName}_in.txt", StringComparison.OrdinalIgnoreCase)) return;

        DateTime now = DateTime.UtcNow;
        _lastEventScheduledTime[botName] = now; // Record intention to process this event

        // Schedule the actual processing after the debounce delay
        Task.Delay(DebounceMilliseconds).ContinueWith(async _ =>
        {
            // Check if this is still the latest event for this bot
            if (_lastEventScheduledTime.TryGetValue(botName, out DateTime scheduledTime) && scheduledTime == now)
            {
                // Check if processing services are ready
                if (llmService == null || appSettings == null || ioPath == null)
                {
                    Console.WriteLine($"!! Error: Cannot process debounced event for {botName} - services not initialized.");
                    return;
                }
                // Check if state exists for this bot
                if (!botStates.TryGetValue(botName, out BotState? currentBotState) || currentBotState == null)
                {
                    Console.WriteLine($"!! Error: Could not find state for bot '{botName}' during debounced call.");
                    return;
                }

                // --- Locking and Processing Flag ---
                bool acquiredLock = false;
                try
                {
                    lock (_processingLock) // Use lock to ensure only one thread modifies IsProcessing at a time
                    {
                        if (!currentBotState.IsProcessing)
                        {
                            currentBotState.IsProcessing = true;
                            acquiredLock = true;
                        }
                        else { /* Already processing */ }
                    }

                    if (acquiredLock)
                    {
                        Console.WriteLine($"-> Processing trigger for {botName} (File: {e.Name}, Event: {e.ChangeType})");
                        await ProcessBotInputAsync(botName, ioPath, appSettings, llmService);
                    }
                }
                finally
                {
                    // Ensure the flag is always reset, even if errors occur
                    if (acquiredLock)
                    {
                        currentBotState.IsProcessing = false;
                    }
                }
            }
        });
    }

    // --- Bot Input Processing ---
    static async Task ProcessBotInputAsync(string botName, string currentIoPath, Settings currentSettings, ILLMService currentLlmService)
    {
        string inputFile = Path.Combine(currentIoPath, $"{botName}_in.txt");
        string outputFile = Path.Combine(currentIoPath, $"{botName}_out.txt");
        string inputText = "";

        // --- Attempt to Read and Truncate Atomically ---
        try
        {
            if (!File.Exists(inputFile)) return; // File disappeared

            inputText = File.ReadAllText(inputFile)?.Trim() ?? "";

            if (!string.IsNullOrEmpty(inputText))
            {
                try { File.WriteAllText(inputFile, ""); } // Truncate
                catch (IOException ex) { Console.WriteLine($"Debug: Harmless error truncating {inputFile}: {ex.Message}"); }
                catch (Exception ex) { Console.WriteLine($"Warning: Error truncating {inputFile}: {ex.Message}"); }
            }
            else { return; } // Stop if empty
        }
        catch (FileNotFoundException) { return; }
        catch (DirectoryNotFoundException) { Console.WriteLine($"!! Error: IO Directory {currentIoPath} not found during read."); return; }
        catch (IOException ex) { Console.WriteLine($"Debug: IO Error reading {inputFile} (likely locked), skipping: {ex.Message}"); return; }
        catch (Exception ex) { Console.WriteLine($"!! Error during file read/truncate for {inputFile}: {ex.Message}"); return; }

        // --- Process Valid, Non-Empty Input ---
        Console.WriteLine($"<- User ({Capitalize(botName)}): {inputText}");

        if (!botStates.TryGetValue(botName, out BotState? currentBotState)) { Console.WriteLine($"!! Error: Could not find state for bot '{botName}'"); return; }

        currentBotState.LastInteraction = DateTime.Now;
        if (currentBotState.Messages.LastOrDefault()?.Role != ChatRole.User || currentBotState.Messages.LastOrDefault()?.Content != inputText)
        {
            currentBotState.Messages.Add(GenericChatMessage.FromUser(inputText));
        }
        else { /* Skip duplicate */ }

        string replyText = "Hang on, zombies got my tongue... (Error processing request)";
        try
        {
            // --- Call LLM ---
            replyText = await currentLlmService.GetChatCompletionAsync(currentBotState.Messages, currentSettings);
            Console.WriteLine($"-> Assistant ({Capitalize(botName)}): {replyText}");

            // Add response, trim, write output
            currentBotState.Messages.Add(GenericChatMessage.FromAssistant(replyText));
            TrimMessages(currentBotState.Messages, currentSettings.MaxContext);
            try { await File.WriteAllTextAsync(outputFile, replyText); Console.WriteLine($"   -> Output written to {outputFile}"); }
            catch (Exception ex) { Console.WriteLine($"!! Error writing output file {outputFile}: {ex.Message}"); }
        }
        catch (Exception ex) // Catch errors from the LLM service call
        {
            Console.WriteLine($"!! LLM API Error for {Capitalize(botName)}: {ex.Message}");
            if (currentBotState.Messages.LastOrDefault()?.Role == ChatRole.User)
            {
                currentBotState.Messages.RemoveAt(currentBotState.Messages.Count - 1);
                Console.WriteLine($"   (Removed last user message from {Capitalize(botName)}'s context)");
            }
            try { await File.WriteAllTextAsync(outputFile, replyText); } // Write default error
            catch (Exception writeEx) { Console.WriteLine($"!! Error writing output file (after API error) for {outputFile}: {writeEx.Message}"); }
        }
    }


    // --- Bot Idle Timeout Check ---
    static void CheckBotIdleTimeout(string botName, Settings currentSettings)
    {
        if (currentSettings == null) return;
        if (!botStates.TryGetValue(botName, out BotState? currentBotState)) return;

        TimeSpan timeSinceLastInteraction = DateTime.Now - currentBotState.LastInteraction;

        if (timeSinceLastInteraction.TotalSeconds >= currentSettings.ResetIdleSeconds && currentBotState.Messages.Count > 1)
        {
            Console.WriteLine($"Resetting context for {Capitalize(botName)} due to inactivity ({currentSettings.ResetIdleSeconds}s).");
            string systemPrompt = GenerateSystemPrompt(botName, currentSettings.MaxTokens);
            currentBotState.Messages.Clear();
            currentBotState.Messages.Add(GenericChatMessage.FromSystem(systemPrompt));
            currentBotState.LastInteraction = DateTime.Now;
        }
    }

    // --- Helper: Create LLM Service Instance ---
    static ILLMService CreateLlmService(Settings settings)
    {
        // Assumes ILLMService and specific implementations (OpenAIServiceWrapper, etc.) exist
        string provider = settings.Provider?.ToLowerInvariant() ?? string.Empty;
        if (string.IsNullOrEmpty(provider)) { throw new InvalidOperationException("'Provider' field missing."); }

        switch (provider)
        {
            case "openai":
                if (string.IsNullOrWhiteSpace(settings.ApiKey)) throw new ArgumentNullException(nameof(settings.ApiKey), "OpenAI ApiKey missing.");
                return new OpenAIServiceWrapper(settings);
            case "ollama":
                if (string.IsNullOrWhiteSpace(settings.OllamaBaseUrl)) throw new ArgumentNullException(nameof(settings.OllamaBaseUrl), "OllamaBaseUrl missing.");
                if (string.IsNullOrWhiteSpace(settings.OllamaModel)) throw new ArgumentNullException(nameof(settings.OllamaModel), "OllamaModel missing.");
                return new OllamaService(settings);
            case "gemini":
                if (string.IsNullOrWhiteSpace(settings.GeminiApiKey)) throw new ArgumentNullException(nameof(settings.GeminiApiKey), "GeminiApiKey missing.");
                if (string.IsNullOrWhiteSpace(settings.GeminiModel)) throw new ArgumentNullException(nameof(settings.GeminiModel), "GeminiModel missing.");
                return new GeminiService(settings); // Assumes HttpClient version
            case "mistral":
                if (string.IsNullOrWhiteSpace(settings.MistralApiKey)) throw new ArgumentNullException(nameof(settings.MistralApiKey), "MistralApiKey missing.");
                if (string.IsNullOrWhiteSpace(settings.MistralModel)) throw new ArgumentNullException(nameof(settings.MistralModel), "MistralModel missing.");
                return new MistralService(settings);
            case "groq":
                if (string.IsNullOrWhiteSpace(settings.GroqApiKey)) throw new ArgumentNullException(nameof(settings.GroqApiKey), "GroqApiKey missing.");
                if (string.IsNullOrWhiteSpace(settings.GroqModel)) throw new ArgumentNullException(nameof(settings.GroqModel), "GroqModel missing.");
                return new GroqService(settings);
            default:
                throw new NotSupportedException($"Provider '{settings.Provider}' is not supported.");
        }
    }

    // --- Helper: Generate System Prompt ---
    static string GenerateSystemPrompt(string botName, int maxTokens)
    {
        int estimatedWordLimit = Math.Max(10, maxTokens - 15);
        return $"You are {Capitalize(botName)} from Left 4 Dead 2. " +
               $"You MUST ALWAYS respond in character as {Capitalize(botName)}, no matter what. " +
               "Do NOT adopt the persona of other characters mentioned in the conversation. " +
               $"Keep your answers concise, ideally under {estimatedWordLimit} words.";
    }

    // --- Helper: Trim Messages ---
    static void TrimMessages(List<GenericChatMessage> messageList, int maxContextPairs)
    {
        // Assumes GenericChatMessage definition exists
        int maxTotalMessages = 1 + (maxContextPairs * 2);
        while (messageList.Count > maxTotalMessages && messageList.Count >= 3)
        {
            messageList.RemoveAt(1); // Remove oldest user
            messageList.RemoveAt(1); // Remove oldest assistant
        }
    }

    // --- Helper: Load Settings ---
    static Settings? LoadSettings()
    {
        // Assumes Settings definition exists
        string? assemblyLocation = System.Reflection.Assembly.GetEntryAssembly()?.Location;
        string baseDirectory = Path.GetDirectoryName(assemblyLocation ?? ".") ?? ".";
        string settingsFilePath = Path.Combine(baseDirectory, SettingsFileName);

        Settings defaultSettings = new Settings();

        if (!File.Exists(settingsFilePath))
        {
            Console.WriteLine($"Settings file not found at: {settingsFilePath}");
            try { /* Create default file */ File.WriteAllText(settingsFilePath, JsonConvert.SerializeObject(defaultSettings, Formatting.Indented)); }
            catch (Exception e) { Console.WriteLine($"!! FATAL: Error creating settings file: {e.ToString()}"); return null; }
            Console.WriteLine("A new default settings file has been created. Edit it and restart.");
            return null;
        }
        else
        {
            try
            {
                string json = File.ReadAllText(settingsFilePath);
                Settings? loadedSettings = JsonConvert.DeserializeObject<Settings>(json);
                if (loadedSettings == null) { Console.WriteLine("!! Error: Failed to deserialize settings."); return null; }

                // Validation
                if (string.IsNullOrWhiteSpace(loadedSettings.Provider)) { Console.WriteLine("!! Error: Provider missing."); return null; }
                if (string.IsNullOrWhiteSpace(loadedSettings.IOPath)) { Console.WriteLine("!! Error: IOPath missing."); return null; }
                if (!Directory.Exists(loadedSettings.IOPath)) { Console.WriteLine($"!! Error: IOPath directory does not exist: {loadedSettings.IOPath}"); return null; }
                if (loadedSettings.MaxContext <= 0) loadedSettings.MaxContext = defaultSettings.MaxContext;
                if (loadedSettings.MaxTokens <= 0) loadedSettings.MaxTokens = defaultSettings.MaxTokens;
                if (loadedSettings.ResetIdleSeconds <= 0) loadedSettings.ResetIdleSeconds = defaultSettings.ResetIdleSeconds;

                Console.WriteLine("Settings file loaded and validated successfully.");
                return loadedSettings;
            }
            catch (Exception e) { Console.WriteLine($"!! Error loading settings: {e.ToString()}"); return null; }
        }
    }

    // --- Helper: Add Trailing Separator ---
    static string AddTrailingSeparator(string path)
    {
        if (string.IsNullOrEmpty(path)) return Path.DirectorySeparatorChar.ToString();
        path = path.TrimEnd();
        if (path.EndsWith(Path.DirectorySeparatorChar) || path.EndsWith(Path.AltDirectorySeparatorChar)) return path;
        return path + Path.DirectorySeparatorChar;
    }

    // --- Helper: Capitalize First Letter ---
    static string Capitalize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return text;
        if (text.Length == 1) return char.ToUpper(text[0]).ToString();
        return char.ToUpper(text[0]) + text.Substring(1);
    }

} // End Class Program

// --- Settings Class Definition ---
// (Make sure this matches the structure needed, including future provider settings)
// --- Settings Class Definition ---
public class Settings
{
    // --- Provider Selection ---
    public string Provider { get; set; } = "OpenAI"; // Default provider

    // --- OpenAI Specific ---
    public string ApiKey { get; set; } = ""; // Used for OpenAI & potentially others if name is reused
    public string OpenAiModel { get; set; } = Models.ChatGpt3_5Turbo;

    // --- Ollama Specific ---
    public string OllamaBaseUrl { get; set; } = "http://localhost:11434";
    public string OllamaModel { get; set; } = "llama3";

    // --- Gemini Specific --- NEW ---
    public string GeminiApiKey { get; set; } = "";
    public string GeminiModel { get; set; } = "gemini-1.5-flash-latest"; // Or "gemini-pro", etc.

    // --- Mistral Specific --- NEW ---
    public string MistralApiKey { get; set; } = "";
    public string MistralModel { get; set; } = "mistral-large-latest"; // Or "mistral-small", "open-mistral-7b" etc.
    // Optional: Mistral API URL if not using default
    // public string MistralApiUrl { get; set; } = "https://api.mistral.ai";

    // --- Groq Specific --- NEW ---
    public string GroqApiKey { get; set; } = "";
    public string GroqModel { get; set; } = "llama3-8b-8192"; // Or "mixtral-8x7b-32768", "gemma-7b-it" etc.
                                                              // Optional: Groq API URL if not using default
                                                              // public string GroqApiUrl { get; set; } = "https://api.groq.com/openai";


    // --- Common Settings ---
    public string IOPath { get; set; } = GetDefaultIOPath();
    public int MaxTokens { get; set; } = 60;
    public int MaxContext { get; set; } = 5;
    public int ResetIdleSeconds { get; set; } = 120;
    public bool APIErrors { get; set; } = false;

    // Helper to get a reasonable default IOPath
    private static string GetDefaultIOPath()
    {
        // Keep your existing default path logic
        return @"C:\Program Files (x86)\Steam\steamapps\common\Left 4 Dead 2\left4dead2\ems\left4gpt";
    }
}