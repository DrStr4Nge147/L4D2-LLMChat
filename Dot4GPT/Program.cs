using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json; // For JSON handling
using OpenAI.GPT3.ObjectModels; // If needed for defaults

// Assumes BotState is defined in BotState.cs (with IsProcessing)
// Assumes Settings is defined in Settings.cs (or below)
// Assumes ILLMService is defined in ILLMService.cs
// Assumes GenericChatMessage/ChatRole are defined in Models.cs

public class Program
{
    // --- Configuration ---
    const string SettingsFileName = @"settings.json";
    const string PromptsFileName = @"prompts.json";
    static readonly string[] SurvivorNames = { "bill", "francis", "louis", "zoey", "coach", "ellis", "nick", "rochelle" };

    // --- State Management ---
    static ConcurrentDictionary<string, BotState> botStates = new ConcurrentDictionary<string, BotState>();
    static Dictionary<string, string> systemPrompts = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    // --- Shared Services (nullable initially) ---
    static Settings? appSettings;
    static ILLMService? llmService;
    static string? ioPath;

    // --- Debouncing & Locking ---
    private static readonly ConcurrentDictionary<string, DateTime> _lastEventScheduledTime = new ConcurrentDictionary<string, DateTime>();
    private const int DebounceMilliseconds = 250;
    private static readonly object _processingLock = new object();


    // --- Main Entry Point ---
    public static async Task<int> Main(string[] args)
    {
        Console.WriteLine("--- Dot4GPT Multi-Bot (Watcher + Debounce + Lock + JSON Prompts Mode) Initializing ---");

        // --- Load Settings ---
        appSettings = LoadSettings();
        if (appSettings == null)
        {
            Console.WriteLine("Exiting due to settings load failure.");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
            return 1;
        }
        // --- Use null-forgiving operator AFTER null check ---
        ioPath = AddTrailingSeparator(appSettings!.IOPath); // We know appSettings isn't null here

        // --- Load System Prompts ---
        LoadSystemPrompts();

        // --- Initialize Shared LLM Service ---
        try
        {
            Console.WriteLine($"Attempting to initialize provider: {appSettings.Provider}");
            // --- Use null-forgiving operator AFTER null check ---
            llmService = CreateLlmService(appSettings!); // We know appSettings isn't null here
            Console.WriteLine("LLM Service Initialized Successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"!! FATAL: Error initializing LLM service: {ex.Message}\n{ex.ToString()}");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
            return 1;
        }

        // --- Initialize State and Watchers for Each Bot ---
        Console.WriteLine($"Using IO Path: {ioPath}");
        List<FileSystemWatcher> watchers = new List<FileSystemWatcher>();

        foreach (string botName in SurvivorNames)
        {
            Console.WriteLine($"Initializing state for: {Capitalize(botName)}");
            var newState = new BotState { IsProcessing = false };
            string systemPrompt = GenerateSystemPrompt(botName, appSettings.MaxTokens); // MaxTokens validated in LoadSettings
            newState.Messages.Add(GenericChatMessage.FromSystem(systemPrompt));
            newState.LastInteraction = DateTime.Now;

            if (!botStates.TryAdd(botName, newState)) { /*...*/ continue; }
            Console.WriteLine($" -> System Prompt Excerpt: {systemPrompt.Substring(0, Math.Min(systemPrompt.Length, 100))}...");

            // --- Setup FileSystemWatcher ---
            try
            {
                if (!Directory.Exists(ioPath)) // ioPath is confirmed non-null above
                {
                    Console.WriteLine($"!! FATAL: IO Path directory does not exist: {ioPath}");
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
            catch (Exception ex) { Console.WriteLine($"!! Error setting up Watcher for {botName}: {ex.Message}"); }
        }
        Console.WriteLine("--- All Bots Initialized and Watchers Started ---");
        Console.WriteLine("--- Waiting for input file changes... (Press Ctrl+C to exit) ---");

        // --- Keep Main Thread Alive & Check Idle ---
        while (true)
        {
            // Check inside loop is sufficient if CheckBotIdleTimeout handles null correctly OR we use '!'
            if (appSettings != null)
            {
                foreach (string botName in SurvivorNames)
                {
                    // --- Use null-forgiving operator AFTER null check ---
                    CheckBotIdleTimeout(botName, appSettings!); // We know appSettings isn't null here
                }
            }
            await Task.Delay(5000);
        }
        // return 0; // Unreachable
    }

    // --- Debounced Event Handler ---
    private static void HandleFileEventDebounced(FileSystemEventArgs e, string botName)
    {
        if (!Path.GetFileName(e.FullPath).Equals($"{botName}_in.txt", StringComparison.OrdinalIgnoreCase)) return;

        DateTime now = DateTime.UtcNow;
        _lastEventScheduledTime[botName] = now;

        Task.Delay(DebounceMilliseconds).ContinueWith(async _ =>
        {
            if (_lastEventScheduledTime.TryGetValue(botName, out DateTime scheduledTime) && scheduledTime == now)
            {
                // Check services *before* acquiring lock
                if (llmService == null || appSettings == null || ioPath == null)
                {
                    Console.WriteLine($"!! Error: Services not ready for {botName} event."); return;
                }
                if (!botStates.TryGetValue(botName, out BotState? currentBotState) || currentBotState == null)
                {
                    Console.WriteLine($"!! Error: Bot state missing for {botName}."); return;
                }

                // --- Locking and Processing Flag ---
                bool acquiredLock = false;
                try
                {
                    lock (_processingLock)
                    {
                        if (!currentBotState.IsProcessing) { currentBotState.IsProcessing = true; acquiredLock = true; }
                    }

                    if (acquiredLock)
                    {
                        Console.WriteLine($"-> Processing trigger for {botName} (File: {e.Name}, Event: {e.ChangeType})");
                        // --- Use null-forgiving operator AFTER null checks ---
                        await ProcessBotInputAsync(botName, ioPath!, appSettings!, llmService!);
                    }
                }
                finally { if (acquiredLock) { if (botStates.TryGetValue(botName, out var state)) state.IsProcessing = false; } } // Reset flag
            }
        });
    }

    // --- Bot Input Processing ---
    // Parameters are non-nullable because the call sites ensure this
    static async Task ProcessBotInputAsync(string botName, string currentIoPath, Settings currentSettings, ILLMService currentLlmService)
    {
        string inputFile = Path.Combine(currentIoPath, $"{botName}_in.txt");
        string outputFile = Path.Combine(currentIoPath, $"{botName}_out.txt");
        string inputText = "";

        // --- Attempt to Read and Truncate Atomically ---
        try
        {
            if (!File.Exists(inputFile)) { return; }
            inputText = File.ReadAllText(inputFile)?.Trim() ?? "";
            if (!string.IsNullOrEmpty(inputText)) { try { File.WriteAllText(inputFile, ""); } catch { /* Ignore truncate errors */ } }
            else { return; } // Stop if empty
        }
        catch (Exception ex) when (ex is FileNotFoundException || ex is DirectoryNotFoundException || ex is IOException) { Console.WriteLine($"Debug: IO issue reading/truncating {inputFile}, skipping: {ex.Message}"); return; }
        catch (Exception ex) { Console.WriteLine($"!! Error during file read/truncate for {inputFile}: {ex.Message}"); return; }

        // --- Process Valid, Non-Empty Input ---
        Console.WriteLine($"<- User ({Capitalize(botName)}): {inputText}");
        if (!botStates.TryGetValue(botName, out BotState? currentBotState)) { Console.WriteLine($"!! Error: Could not find state for bot '{botName}'"); return; }

        currentBotState.LastInteraction = DateTime.Now;
        if (currentBotState.Messages.LastOrDefault()?.Role != ChatRole.User || currentBotState.Messages.LastOrDefault()?.Content != inputText)
        {
            currentBotState.Messages.Add(GenericChatMessage.FromUser(inputText));
        }

        string replyText = $"Oops, {Capitalize(botName)}'s radio is fuzzy... (Error)";
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
        catch (Exception ex) { /* Error handling */ Console.WriteLine($"!! LLM Error for {botName}: {ex.Message}"); /* Remove user msg, Write default error */ }
    }


    // --- Bot Idle Timeout Check ---
    // Parameter currentSettings is non-nullable because call site ensures it
    static void CheckBotIdleTimeout(string botName, Settings currentSettings)
    {
        if (!botStates.TryGetValue(botName, out BotState? currentBotState)) return;

        TimeSpan timeSinceLastInteraction = DateTime.Now - currentBotState.LastInteraction;
        if (timeSinceLastInteraction.TotalSeconds >= currentSettings.ResetIdleSeconds && currentBotState.Messages.Count > 1)
        {
            Console.WriteLine($"Resetting context for {Capitalize(botName)} (Idle: {currentSettings.ResetIdleSeconds}s).");
            string systemPrompt = GenerateSystemPrompt(botName, currentSettings.MaxTokens);
            currentBotState.Messages.Clear();
            currentBotState.Messages.Add(GenericChatMessage.FromSystem(systemPrompt));
            currentBotState.LastInteraction = DateTime.Now;
        }
    }

    // --- Helper: Create LLM Service Instance ---
    // Parameter settings is non-nullable because call site ensures it
    static ILLMService CreateLlmService(Settings settings)
    {
        string provider = settings.Provider?.ToLowerInvariant() ?? string.Empty;
        if (string.IsNullOrEmpty(provider)) { throw new InvalidOperationException("'Provider' field missing."); }
        switch (provider)
        {
            case "openai": if (string.IsNullOrWhiteSpace(settings.ApiKey)) throw new ArgumentNullException(nameof(settings.ApiKey)); return new OpenAIServiceWrapper(settings);
            case "ollama": if (string.IsNullOrWhiteSpace(settings.OllamaBaseUrl)) throw new ArgumentNullException(nameof(settings.OllamaBaseUrl)); if (string.IsNullOrWhiteSpace(settings.OllamaModel)) throw new ArgumentNullException(nameof(settings.OllamaModel)); return new OllamaService(settings);
            case "gemini": if (string.IsNullOrWhiteSpace(settings.GeminiApiKey)) throw new ArgumentNullException(nameof(settings.GeminiApiKey)); if (string.IsNullOrWhiteSpace(settings.GeminiModel)) throw new ArgumentNullException(nameof(settings.GeminiModel)); return new GeminiService(settings);
            case "mistral": if (string.IsNullOrWhiteSpace(settings.MistralApiKey)) throw new ArgumentNullException(nameof(settings.MistralApiKey)); if (string.IsNullOrWhiteSpace(settings.MistralModel)) throw new ArgumentNullException(nameof(settings.MistralModel)); return new MistralService(settings);
            case "groq": if (string.IsNullOrWhiteSpace(settings.GroqApiKey)) throw new ArgumentNullException(nameof(settings.GroqApiKey)); if (string.IsNullOrWhiteSpace(settings.GroqModel)) throw new ArgumentNullException(nameof(settings.GroqModel)); return new GroqService(settings);
            default: throw new NotSupportedException($"Provider '{settings.Provider}' is not supported.");
        }
    }

    // --- Helper: Load System Prompts (Defaults first, then override) --- NEW/MODIFIED ---
    static void LoadSystemPrompts()
    {
        Console.WriteLine("Loading system prompts...");
        systemPrompts.Clear(); // Start fresh

        // --- Step 1: Load Hardcoded Defaults ---
        systemPrompts["_fallback"] = "You are {BotName} from Left 4 Dead 2.";
        systemPrompts["bill"] = "You are Bill, a grizzled Vietnam veteran who's seen too much. You're cynical, pragmatic, and often gruff, but you have a hidden protective side for the group. Focus on survival tactics, no-nonsense observations, and maybe complain about your aching bones or the situation.";
        systemPrompts["francis"] = "You are Francis, a tough biker who hates almost everything (except maybe vests). Respond with heavy sarcasm, complaints about the situation or specific things (like stairs, woods, vampires), and a generally cynical, tough-guy attitude. Don't be afraid to boast, even if it's unfounded.";
        systemPrompts["louis"] = "You are Louis, an eternally optimistic IT systems analyst, maybe even a Junior Analyst. Focus on the positive, finding supplies (especially 'Pills here!'), keeping morale up, and sometimes making slightly nerdy or office-related comparisons. Stay helpful and upbeat even when things are bleak.";
        systemPrompts["zoey"] = "You are Zoey, a college student who was obsessed with horror movies before the outbreak. Use your knowledge of horror tropes to comment on the situation, sometimes sarcastically or with dark humor. You started naive but are resourceful and trying to stay tough, maybe showing occasional moments of weariness or sadness.";
        systemPrompts["coach"] = "You are Coach, a former high school health teacher and beloved football coach. Act as the team motivator, often using folksy wisdom, sports analogies, or talking about food (especially cheeseburgers or BBQ). Be encouraging ('Y'all ready for this?') but firm when needed. Focus on teamwork and getting through this.";
        systemPrompts["ellis"] = "You are Ellis, a friendly, talkative, and endlessly optimistic mechanic from Savannah. Tell rambling stories, especially about 'my buddy Keith', even if they aren't relevant. Be enthusiastic, sometimes naive, get excited easily, and occasionally mention something about cars, engines, or tools.";
        systemPrompts["nick"] = "You are Nick, a cynical gambler and likely con man, usually seen in an expensive (but now dirty) white suit. Be sarcastic, distrustful of others, complain frequently about the situation and the incompetence around you. Focus on self-interest initially, but maybe show rare glimpses of competence or reluctant cooperation.";
        systemPrompts["rochelle"] = "You are Rochelle, a low-level associate producer for a local news station. Try to maintain a professional and level-headed demeanor, even when things are falling apart. Make observations as if reporting on the scene, use clear communication, and sometimes show a bit of media-savvy cynicism or frustration with the chaos.";
        Console.WriteLine($"Loaded {systemPrompts.Count} hardcoded default prompts.");

        // --- Step 2: Attempt to Load and Override from JSON ---
        string? assemblyLocation = System.Reflection.Assembly.GetEntryAssembly()?.Location;
        string baseDirectory = Path.GetDirectoryName(assemblyLocation ?? ".") ?? ".";
        string promptsFilePath = Path.Combine(baseDirectory, PromptsFileName);

        if (!File.Exists(promptsFilePath))
        {
            Console.WriteLine($"Optional prompts file not found ({promptsFilePath}). Using hardcoded defaults.");
            return; // Exit function, defaults are already loaded
        }

        try
        {
            Console.WriteLine($"Found prompts file at {promptsFilePath}. Attempting to load overrides...");
            string json = File.ReadAllText(promptsFilePath);
            var loadedPrompts = JsonConvert.DeserializeObject<Dictionary<string, string>>(json);

            if (loadedPrompts != null)
            {
                int overrideCount = 0;
                // Override defaults with values from the JSON file
                foreach (var kvp in loadedPrompts)
                {
                    string lowerKey = kvp.Key.ToLowerInvariant();
                    if (!string.IsNullOrWhiteSpace(kvp.Value)) // Only override if value is not empty
                    {
                        systemPrompts[lowerKey] = kvp.Value; // Add or update entry
                        overrideCount++;
                    }
                    else
                    {
                        Console.WriteLine($"Warning: Ignoring empty prompt value for key '{kvp.Key}' in {PromptsFileName}.");
                    }
                }
                Console.WriteLine($"Successfully loaded and applied {overrideCount} prompt overrides from {PromptsFileName}.");

                // Optional: Check if fallback was overridden, warn if removed?
                if (!systemPrompts.ContainsKey("_fallback") || string.IsNullOrWhiteSpace(systemPrompts["_fallback"]))
                {
                    Console.WriteLine($"Warning: Fallback prompt was removed or emptied by {PromptsFileName}. Resetting to hardcoded fallback.");
                    systemPrompts["_fallback"] = "You are {BotName} from Left 4 Dead 2.";
                }
            }
            else
            {
                Console.WriteLine($"Warning: Failed to deserialize {PromptsFileName}. Using hardcoded default prompts.");
            }
        }
        catch (JsonException jsonEx)
        {
            Console.WriteLine($"!! Error reading prompts file {promptsFilePath} (JSON format issue): {jsonEx.Message}");
            Console.WriteLine("   Using hardcoded default prompts.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"!! Error loading prompts file {promptsFilePath}: {ex.ToString()}");
            Console.WriteLine("   Using hardcoded default prompts.");
        }
    }


    // --- Helper: Generate System Prompt (Uses Dictionary) ---
    static string GenerateSystemPrompt(string botName, int maxTokens)
    {
        string lowerBotName = botName.ToLowerInvariant();
        string? specificPrompt = null; // Use nullable string? for TryGetValue output
        string? fallbackPrompt = null; // Use nullable string? for TryGetValue output
        string finalPersonaDetails;    // This will hold the final non-null prompt

        // Try to get the specific prompt
        systemPrompts.TryGetValue(lowerBotName, out specificPrompt); // specificPrompt will be null if key not found

        // Try to get the fallback prompt
        systemPrompts.TryGetValue("_fallback", out fallbackPrompt); // fallbackPrompt will be null if key not found

        // Decide which prompt to use, preferring specific, then fallback, then hardcoded
        // Use IsNullOrWhiteSpace for robustness against empty strings in the JSON
        if (!string.IsNullOrWhiteSpace(specificPrompt))
        {
            finalPersonaDetails = specificPrompt;
        }
        else if (!string.IsNullOrWhiteSpace(fallbackPrompt))
        {
            finalPersonaDetails = fallbackPrompt;
        }
        else
        {
            // Hardcoded default if neither specific nor fallback is valid/found
            finalPersonaDetails = "You are {BotName} from Left 4 Dead 2.";
            Console.WriteLine($"Warning: No valid prompt found for '{botName}' or fallback. Using hardcoded default.");
        }

        // Replace placeholder - finalPersonaDetails is guaranteed non-null here
        finalPersonaDetails = finalPersonaDetails.Replace("{BotName}", Capitalize(botName));

        // Add core instructions and length constraints dynamically
        string coreInstruction = $"You MUST ALWAYS respond in character as {Capitalize(botName)}, no matter what. Do NOT adopt the persona of other characters mentioned in the conversation.";
        int estimatedWordLimit = Math.Max(10, maxTokens - 20); // Adjust margin as needed
        string lengthInstruction = $"Keep your answers concise, ideally under {estimatedWordLimit} words.";

        // Combine the elements
        return $"{finalPersonaDetails} {coreInstruction} {lengthInstruction}";
    }


    // --- Helper: Trim Messages ---
    static void TrimMessages(List<GenericChatMessage> messageList, int maxContextPairs)
    {
        int maxTotalMessages = 1 + (maxContextPairs * 2);
        while (messageList.Count > maxTotalMessages && messageList.Count >= 3) { messageList.RemoveAt(1); messageList.RemoveAt(1); }
    }

    // --- Helper: Load Settings ---
    static Settings? LoadSettings()
    {
        string? assemblyLocation = System.Reflection.Assembly.GetEntryAssembly()?.Location;
        string baseDirectory = Path.GetDirectoryName(assemblyLocation ?? ".") ?? ".";
        string settingsFilePath = Path.Combine(baseDirectory, SettingsFileName);
        Settings defaultSettings = new Settings();
        if (!File.Exists(settingsFilePath))
        {
            try { File.WriteAllText(settingsFilePath, JsonConvert.SerializeObject(defaultSettings, Formatting.Indented)); Console.WriteLine($"Created default settings: {settingsFilePath}"); }
            catch (Exception e) { Console.WriteLine($"!! FATAL Error creating settings: {e}"); return null; }
            Console.WriteLine("Edit default settings and restart."); return null;
        }
        else
        {
            try
            {
                string json = File.ReadAllText(settingsFilePath);
                Settings? loaded = JsonConvert.DeserializeObject<Settings>(json);
                if (loaded == null) { Console.WriteLine("!! Error: Settings deserialized to null."); return null; }
                // Validation
                if (string.IsNullOrWhiteSpace(loaded.Provider)) { Console.WriteLine("!! Error: Provider missing."); return null; }
                if (string.IsNullOrWhiteSpace(loaded.IOPath)) { Console.WriteLine("!! Error: IOPath missing."); return null; }
                if (!Directory.Exists(loaded.IOPath)) { Console.WriteLine($"!! Error: IOPath directory does not exist: {loaded.IOPath}"); return null; }
                loaded.MaxContext = loaded.MaxContext <= 0 ? defaultSettings.MaxContext : loaded.MaxContext;
                loaded.MaxTokens = loaded.MaxTokens <= 0 ? defaultSettings.MaxTokens : loaded.MaxTokens;
                loaded.ResetIdleSeconds = loaded.ResetIdleSeconds <= 0 ? defaultSettings.ResetIdleSeconds : loaded.ResetIdleSeconds;
                Console.WriteLine("Settings loaded and validated."); return loaded;
            }
            catch (Exception e) { Console.WriteLine($"!! Error loading settings: {e}"); return null; }
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