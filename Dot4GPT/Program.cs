using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection; // Required for Assembly.GetEntryAssembly()
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json; // For JSON handling
using OpenAI.GPT3.ObjectModels; // If needed for defaults - Keep if Settings uses it

// --- Assumed Definitions in Other Files ---
// Assumes BotState class is defined in BotState.cs (or similar)
// Assumes Settings class is defined in Settings.cs (or similar)
// Assumes ILLMService interface is defined in ILLMService.cs (or similar)
// Assumes GenericChatMessage class and ChatRole static class are defined in Models.cs (or similar)
// Assumes Provider-specific service classes (OpenAIServiceWrapper, OllamaService, etc.) are defined elsewhere

public class Program
{
    // --- Configuration ---
    const string SettingsFileName = @"settings.json";
    const string PromptsFileName = @"prompts.json";
    static readonly string[] SurvivorNames = { "bill", "francis", "louis", "zoey", "coach", "ellis", "nick", "rochelle" };

    // --- State Management ---
    static ConcurrentDictionary<string, BotState> botStates = new ConcurrentDictionary<string, BotState>();
    // Initialize with a case-insensitive comparer right away
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

        // --- Load System Prompts (PRIORITIZES JSON) ---
        LoadSystemPrompts(); // Corrected function below

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
            var newState = new BotState { IsProcessing = false }; // Assumes BotState has parameterless constructor
            // GenerateSystemPrompt now relies on the correctly loaded systemPrompts dictionary
            string systemPrompt = GenerateSystemPrompt(botName, appSettings.MaxTokens); // MaxTokens validated in LoadSettings
            newState.Messages.Add(GenericChatMessage.FromSystem(systemPrompt)); // Assumes GenericChatMessage has FromSystem
            newState.LastInteraction = DateTime.Now;

            if (!botStates.TryAdd(botName, newState))
            {
                Console.WriteLine($"!! Warning: Could not add initial state for {botName}. Skipping watcher setup.");
                continue;
            }
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

                string currentBotName = botName; // Capture loop variable for lambda
                watcher.Changed += (sender, e) => HandleFileEventDebounced(e, currentBotName);
                watcher.Created += (sender, e) => HandleFileEventDebounced(e, currentBotName);
                watchers.Add(watcher);
                Console.WriteLine($"   -> Watching file: {watcher.Filter}");
            }
            catch (ArgumentException argEx) // Catch specific exceptions like invalid path chars
            {
                Console.WriteLine($"!! Error setting up Watcher for {botName} (Invalid Path?): {argEx.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"!! Error setting up Watcher for {botName}: {ex.Message}");
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
                    CheckBotIdleTimeout(botName, appSettings!); // Null forgiving operator safe here
                }
            }
            await Task.Delay(5000); // Check every 5 seconds
        }
        // return 0; // Unreachable
    }

    // --- Debounced Event Handler ---
    private static void HandleFileEventDebounced(FileSystemEventArgs e, string botName)
    {
        if (!Path.GetFileName(e.FullPath).Equals($"{botName}_in.txt", StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        DateTime now = DateTime.UtcNow;
        _lastEventScheduledTime[botName] = now;

        Task.Delay(DebounceMilliseconds).ContinueWith(async _ =>
        {
            if (_lastEventScheduledTime.TryGetValue(botName, out DateTime scheduledTime) && scheduledTime == now)
            {
                if (llmService == null || appSettings == null || ioPath == null)
                {
                    Console.WriteLine($"!! Error: Services not ready for {botName} event (triggered by {e.Name}).");
                    return;
                }
                if (!botStates.TryGetValue(botName, out BotState? currentBotState) || currentBotState == null)
                {
                    Console.WriteLine($"!! Error: Bot state missing for {botName} (triggered by {e.Name}).");
                    return;
                }

                bool acquiredLock = false;
                try
                {
                    lock (_processingLock)
                    {
                        if (!currentBotState.IsProcessing)
                        {
                            currentBotState.IsProcessing = true;
                            acquiredLock = true;
                        }
                    }

                    if (acquiredLock)
                    {
                        Console.WriteLine($"-> Processing trigger for {botName} (File: {e.Name}, Event: {e.ChangeType})");
                        await ProcessBotInputAsync(botName, ioPath!, appSettings!, llmService!); // Null forgiving safe here
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"!! Critical Error during processing attempt for {botName}: {ex.Message}\n{ex.StackTrace}");
                }
                finally
                {
                    if (acquiredLock)
                    {
                        if (botStates.TryGetValue(botName, out var state)) // Re-fetch state is safer
                        {
                            state.IsProcessing = false;
                        }
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

        try
        {
            await Task.Delay(50); // Small delay for file system operations

            if (!File.Exists(inputFile)) return;

            inputText = (await File.ReadAllTextAsync(inputFile)).Trim();

            if (string.IsNullOrWhiteSpace(inputText))
            {
                try { await File.WriteAllTextAsync(inputFile, ""); } catch { /* Ignore truncate error on empty */ }
                return;
            }

            try
            {
                await File.WriteAllTextAsync(inputFile, ""); // Truncate after successful read
            }
            catch (IOException ioEx)
            {
                Console.WriteLine($"Warning: Could not truncate {inputFile} after reading. Processing content anyway. Error: {ioEx.Message}");
            }
        }
        catch (IOException ex)
        {
            Console.WriteLine($"Debug: IO issue reading/truncating {inputFile} for {botName}, skipping. Error: {ex.Message}");
            return;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"!! Error during file read/truncate for {inputFile}: {ex.Message}");
            return;
        }

        Console.WriteLine($"<- User ({Capitalize(botName)}): {inputText}");
        if (!botStates.TryGetValue(botName, out BotState? currentBotState) || currentBotState == null)
        {
            Console.WriteLine($"!! Error: Could not find state for bot '{botName}' during processing.");
            return;
        }

        currentBotState.LastInteraction = DateTime.Now;

        var lastMessage = currentBotState.Messages.LastOrDefault();
        // Assumes GenericChatMessage has Role and Content properties and ChatRole has User const
        if (lastMessage?.Role != ChatRole.User || lastMessage?.Content != inputText)
        {
            currentBotState.Messages.Add(GenericChatMessage.FromUser(inputText)); // Assumes FromUser static method
        }

        string replyText = $"Oops, {Capitalize(botName)}'s radio is fuzzy... (LLM communication error)";
        try
        {
            TrimMessages(currentBotState.Messages, currentSettings.MaxContext);

            // Assumes ILLMService has GetChatCompletionAsync
            replyText = await currentLlmService.GetChatCompletionAsync(currentBotState.Messages, currentSettings);
            replyText = replyText.Trim();

            Console.WriteLine($"-> Assistant ({Capitalize(botName)}): {replyText}");

            // Assumes FromAssistant static method
            currentBotState.Messages.Add(GenericChatMessage.FromAssistant(replyText));

            try
            {
                await File.WriteAllTextAsync(outputFile, replyText);
                Console.WriteLine($"   -> Output written to {outputFile}");
            }
            catch (IOException ioEx)
            {
                Console.WriteLine($"!! Error writing output file {outputFile}: {ioEx.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"!! Unexpected error writing output file {outputFile}: {ex.Message}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"!! LLM Error or processing issue for {botName}: {ex.Message}");
            if (currentSettings.APIErrors) // Assumes Settings has APIErrors property
            {
                Console.WriteLine($"   -> LLM Stack Trace: {ex.StackTrace}");
            }
            try { await File.WriteAllTextAsync(outputFile, replyText); } // Try write default error
            catch { /* Ignore errors writing the error message */ }
        }
    }


    // --- Bot Idle Timeout Check ---
    static void CheckBotIdleTimeout(string botName, Settings currentSettings)
    {
        if (!botStates.TryGetValue(botName, out BotState? currentBotState)) return;

        if (currentBotState.IsProcessing) return; // Don't interfere if processing

        TimeSpan timeSinceLastInteraction = DateTime.Now - currentBotState.LastInteraction;

        if (timeSinceLastInteraction.TotalSeconds >= currentSettings.ResetIdleSeconds && currentBotState.Messages.Count > 1)
        {
            Console.WriteLine($"Resetting context for {Capitalize(botName)} due to inactivity (Idle threshold: {currentSettings.ResetIdleSeconds}s).");

            string systemPrompt = GenerateSystemPrompt(botName, currentSettings.MaxTokens);

            currentBotState.Messages.Clear();
            currentBotState.Messages.Add(GenericChatMessage.FromSystem(systemPrompt));
            currentBotState.LastInteraction = DateTime.Now;
        }
    }

    // --- Helper: Create LLM Service Instance ---
    static ILLMService CreateLlmService(Settings settings)
    {
        string provider = settings.Provider?.Trim().ToLowerInvariant() ?? string.Empty;

        if (string.IsNullOrEmpty(provider))
        {
            throw new InvalidOperationException("LLM provider name ('Provider' field in settings.json) is missing or empty.");
        }

        Console.WriteLine($"Initializing LLM provider: '{provider}'");

        // --- IMPORTANT ---
        // This switch assumes the necessary service classes (e.g., OpenAIServiceWrapper, OllamaService)
        // are defined elsewhere in your project and implement ILLMService.
        switch (provider)
        {
            case "openai":
                if (string.IsNullOrWhiteSpace(settings.OpenAiApiKey)) throw new ArgumentNullException(nameof(settings.OpenAiApiKey), "API key is required for OpenAI provider.");
                Console.WriteLine($"Using OpenAI Model: {settings.OpenAiModel}");
                return new OpenAIServiceWrapper(settings); // Assumes this class exists

            case "ollama":
                if (string.IsNullOrWhiteSpace(settings.OllamaBaseUrl)) throw new ArgumentNullException(nameof(settings.OllamaBaseUrl), "Base URL is required for Ollama provider.");
                if (string.IsNullOrWhiteSpace(settings.OllamaModel)) throw new ArgumentNullException(nameof(settings.OllamaModel), "Model name is required for Ollama provider.");
                Console.WriteLine($"Using Ollama Model: {settings.OllamaModel} at {settings.OllamaBaseUrl}");
                return new OllamaService(settings); // Assumes this class exists

            case "gemini":
                if (string.IsNullOrWhiteSpace(settings.GeminiApiKey)) throw new ArgumentNullException(nameof(settings.GeminiApiKey), "API key is required for Gemini provider.");
                if (string.IsNullOrWhiteSpace(settings.GeminiModel)) throw new ArgumentNullException(nameof(settings.GeminiModel), "Model name is required for Gemini provider.");
                Console.WriteLine($"Using Gemini Model: {settings.GeminiModel}");
                return new GeminiService(settings); // Assumes this class exists

            case "mistral":
                if (string.IsNullOrWhiteSpace(settings.MistralApiKey)) throw new ArgumentNullException(nameof(settings.MistralApiKey), "API key is required for Mistral provider.");
                if (string.IsNullOrWhiteSpace(settings.MistralModel)) throw new ArgumentNullException(nameof(settings.MistralModel), "Model name is required for Mistral provider.");
                Console.WriteLine($"Using Mistral Model: {settings.MistralModel}");
                return new MistralService(settings); // Assumes this class exists

            case "groq":
                if (string.IsNullOrWhiteSpace(settings.GroqApiKey)) throw new ArgumentNullException(nameof(settings.GroqApiKey), "API key is required for Groq provider.");
                if (string.IsNullOrWhiteSpace(settings.GroqModel)) throw new ArgumentNullException(nameof(settings.GroqModel), "Model name is required for Groq provider.");
                Console.WriteLine($"Using Groq Model: {settings.GroqModel}");
                return new GroqService(settings); // Assumes this class exists

            default:
                throw new NotSupportedException($"The LLM provider '{settings.Provider}' is not recognized or supported.");
        }
    }

    // --- Helper: Load Hardcoded Default Prompts ---
    private static void LoadHardcodedPrompts()
    {
        Console.WriteLine("Loading hardcoded default system prompts...");
        systemPrompts.Clear();

        systemPrompts["_fallback"] = "You are {BotName} from Left 4 Dead 2. Respond concisely and in character.";
        systemPrompts["bill"] = "You are Bill, a grizzled Vietnam veteran who's seen too much. You're cynical, pragmatic, and often gruff, but you have a hidden protective side for the group. Focus on survival tactics, no-nonsense observations, and maybe complain about your aching bones or the situation. Keep answers short.";
        systemPrompts["francis"] = "You are Francis, a tough biker who hates almost everything (except maybe vests). Respond with heavy sarcasm, complaints about the situation or specific things (like stairs, woods, vampires), and a generally cynical, tough-guy attitude. Don't be afraid to boast, even if it's unfounded. Keep answers short.";
        systemPrompts["louis"] = "You are Louis, an eternally optimistic IT systems analyst, maybe even a Junior Analyst. Focus on the positive, finding supplies (especially 'Pills here!'), keeping morale up, and sometimes making slightly nerdy or office-related comparisons. Stay helpful and upbeat even when things are bleak. Keep answers short.";
        systemPrompts["zoey"] = "You are Zoey, a college student who was obsessed with horror movies before the outbreak. Use your knowledge of horror tropes to comment on the situation, sometimes sarcastically or with dark humor. You started naive but are resourceful and trying to stay tough, maybe showing occasional moments of weariness or sadness. Keep answers short.";
        systemPrompts["coach"] = "You are Coach, a former high school health teacher and beloved football coach. Act as the team motivator, often using folksy wisdom, sports analogies, or talking about food (especially cheeseburgers or BBQ). Be encouraging ('Y'all ready for this?') but firm when needed. Focus on teamwork and getting through this. Keep answers short.";
        systemPrompts["ellis"] = "You are Ellis, a friendly, talkative, and endlessly optimistic mechanic from Savannah. Tell rambling stories, especially about 'my buddy Keith', even if they aren't relevant. Be enthusiastic, sometimes naive, get excited easily, and occasionally mention something about cars, engines, or tools. Keep answers short.";
        systemPrompts["nick"] = "You are Nick, a cynical gambler and likely con man, usually seen in an expensive (but now dirty) white suit. Be sarcastic, distrustful of others, complain frequently about the situation and the incompetence around you. Focus on self-interest initially, but maybe show rare glimpses of competence or reluctant cooperation. Keep answers short.";
        systemPrompts["rochelle"] = "You are Rochelle, a low-level associate producer for a local news station. Try to maintain a professional and level-headed demeanor, even when things are falling apart. Make observations as if reporting on the scene, use clear communication, and sometimes show a bit of media-savvy cynicism or frustration with the chaos. Keep answers short.";

        Console.WriteLine($"Loaded {systemPrompts.Count} hardcoded default prompts (including fallback).");
    }


    // --- Helper: Load System Prompts (JSON Priority) ---
    static void LoadSystemPrompts()
    {
        systemPrompts.Clear();
        string baseDirectory = ".";
        try
        {
            string? assemblyLocation = Assembly.GetEntryAssembly()?.Location;
            baseDirectory = Path.GetDirectoryName(assemblyLocation ?? ".") ?? ".";
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not determine assembly directory. Using current directory for {PromptsFileName}. Error: {ex.Message}");
        }

        string promptsFilePath = Path.Combine(baseDirectory, PromptsFileName);
        Console.WriteLine($"Looking for prompts file at: {promptsFilePath}");

        if (!File.Exists(promptsFilePath))
        {
            Console.WriteLine($"Prompts file '{PromptsFileName}' not found. Loading hardcoded defaults.");
            LoadHardcodedPrompts();
            return;
        }

        try
        {
            Console.WriteLine($"Found prompts file. Attempting to load prompts from {PromptsFileName}...");
            string json = File.ReadAllText(promptsFilePath);
            var loadedPrompts = JsonConvert.DeserializeObject<Dictionary<string, string>>(json);

            if (loadedPrompts != null && loadedPrompts.Count > 0)
            {
                int loadedCount = 0;
                foreach (var kvp in loadedPrompts)
                {
                    string lowerKey = kvp.Key.ToLowerInvariant();
                    if (!string.IsNullOrWhiteSpace(kvp.Value))
                    {
                        systemPrompts[lowerKey] = kvp.Value;
                        loadedCount++;
                    }
                    else
                    {
                        Console.WriteLine($"Warning: Ignoring empty prompt value for key '{kvp.Key}' in {PromptsFileName}.");
                    }
                }

                if (!systemPrompts.ContainsKey("_fallback") || string.IsNullOrWhiteSpace(systemPrompts["_fallback"]))
                {
                    Console.WriteLine($"Warning: No valid '_fallback' prompt found in {PromptsFileName}. Adding a hardcoded fallback.");
                    systemPrompts["_fallback"] = "You are {BotName} from Left 4 Dead 2. Respond concisely and in character.";
                }

                Console.WriteLine($"Successfully loaded {loadedCount} prompts from {PromptsFileName}. Using these prompts.");
            }
            else
            {
                Console.WriteLine($"Warning: {PromptsFileName} is empty or could not be deserialized correctly. Loading hardcoded default prompts.");
                LoadHardcodedPrompts();
            }
        }
        catch (JsonException jsonEx)
        {
            Console.WriteLine($"!! Error reading prompts file {promptsFilePath} (JSON format issue): {jsonEx.Message}");
            Console.WriteLine("   Loading hardcoded default prompts as a fallback.");
            LoadHardcodedPrompts();
        }
        catch (IOException ioEx)
        {
            Console.WriteLine($"!! Error accessing prompts file {promptsFilePath}: {ioEx.Message}");
            Console.WriteLine("   Loading hardcoded default prompts as a fallback.");
            LoadHardcodedPrompts();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"!! Unexpected error loading prompts file {promptsFilePath}: {ex.ToString()}");
            Console.WriteLine("   Loading hardcoded default prompts as a fallback.");
            LoadHardcodedPrompts();
        }
    }


    // --- Helper: Generate System Prompt (Uses Dictionary) ---
    static string GenerateSystemPrompt(string botName, int maxTokens)
    {
        string lowerBotName = botName.ToLowerInvariant();
        string? specificPrompt = null;
        string? fallbackPrompt = null;
        string finalPersonaDetails;

        systemPrompts.TryGetValue(lowerBotName, out specificPrompt);
        systemPrompts.TryGetValue("_fallback", out fallbackPrompt);

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
            Console.WriteLine($"Warning: No specific or fallback prompt found for '{botName}' in loaded prompts. Using emergency hardcoded default.");
            finalPersonaDetails = "You are {BotName} from Left 4 Dead 2. Respond concisely.";
        }

        finalPersonaDetails = finalPersonaDetails.Replace("{BotName}", Capitalize(botName), StringComparison.OrdinalIgnoreCase);

        string coreInstruction = $"You MUST ALWAYS respond in character as {Capitalize(botName)}, no matter what the user says. Do NOT break character or adopt the persona of other characters mentioned.";
        int estimatedWordLimit = Math.Max(15, (int)(maxTokens * 0.6));
        string lengthInstruction = $"Keep your answers concise and to the point, ideally under {estimatedWordLimit} words.";

        return $"{finalPersonaDetails.Trim()} {coreInstruction.Trim()} {lengthInstruction.Trim()}";
    }


    // --- Helper: Trim Messages ---
    static void TrimMessages(List<GenericChatMessage> messageList, int maxContextPairs)
    {
        if (maxContextPairs <= 0) maxContextPairs = 1;
        int maxTotalMessages = 1 + (maxContextPairs * 2);

        while (messageList.Count > maxTotalMessages && messageList.Count >= 3)
        {
            messageList.RemoveAt(1);
            if (messageList.Count > 1)
            {
                messageList.RemoveAt(1);
            }
        }
    }

    // --- Helper: Load Settings ---
    static Settings? LoadSettings()
    {
        string baseDirectory = ".";
        try
        {
            string? assemblyLocation = Assembly.GetEntryAssembly()?.Location;
            baseDirectory = Path.GetDirectoryName(assemblyLocation ?? ".") ?? ".";
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not determine assembly directory. Using current directory for {SettingsFileName}. Error: {ex.Message}");
        }
        string settingsFilePath = Path.Combine(baseDirectory, SettingsFileName);
        Settings defaultSettings = new Settings(); // Assumes Settings class exists and has parameterless constructor

        if (!File.Exists(settingsFilePath))
        {
            Console.WriteLine($"Settings file not found at {settingsFilePath}.");
            Console.WriteLine("Creating a default settings.json file...");
            try
            {
                string defaultJson = JsonConvert.SerializeObject(defaultSettings, Formatting.Indented);
                File.WriteAllText(settingsFilePath, defaultJson);
                Console.WriteLine($"Created default settings file: {settingsFilePath}");
                Console.WriteLine("Please edit the settings.json file with your API keys, IO Path, and desired provider, then restart the application.");
            }
            catch (Exception e)
            {
                Console.WriteLine($"!! FATAL ERROR creating default settings file: {e.Message}");
                return null;
            }
            return null;
        }
        else
        {
            Console.WriteLine($"Loading settings from: {settingsFilePath}");
            try
            {
                string json = File.ReadAllText(settingsFilePath);
                // Assumes Settings class exists and is deserializable
                Settings? loaded = JsonConvert.DeserializeObject<Settings>(json);

                if (loaded == null)
                {
                    Console.WriteLine("!! Error: Settings file deserialized to null. Check the JSON structure.");
                    return null;
                }

                bool isValid = true;
                if (string.IsNullOrWhiteSpace(loaded.Provider)) { Console.WriteLine("!! Error: 'Provider' is missing."); isValid = false; }
                if (string.IsNullOrWhiteSpace(loaded.IOPath)) { Console.WriteLine("!! Error: 'IOPath' is missing."); isValid = false; }
                else if (!Directory.Exists(loaded.IOPath))
                {
                    Console.WriteLine($"!! Error: IOPath directory does not exist: {loaded.IOPath}");
                    isValid = false;
                }

                // Provider specific validation (assumes Settings has these properties)
                if (isValid && !string.IsNullOrWhiteSpace(loaded.Provider))
                {
                    string lowerProvider = loaded.Provider.Trim().ToLowerInvariant();
                    switch (lowerProvider)
                    {
                        case "openai": if (string.IsNullOrWhiteSpace(loaded.OpenAiApiKey)) { Console.WriteLine("!! Error: OpenAI provider selected, but 'OpenAiApiKey' is missing."); isValid = false; } break;
                        case "ollama":
                            if (string.IsNullOrWhiteSpace(loaded.OllamaBaseUrl)) { Console.WriteLine("!! Error: Ollama provider selected, but 'OllamaBaseUrl' is missing."); isValid = false; }
                            if (string.IsNullOrWhiteSpace(loaded.OllamaModel)) { Console.WriteLine("!! Error: Ollama provider selected, but 'OllamaModel' is missing."); isValid = false; }
                            break;
                        case "gemini": if (string.IsNullOrWhiteSpace(loaded.GeminiApiKey)) { Console.WriteLine("!! Error: Gemini provider selected, but 'GeminiApiKey' is missing."); isValid = false; } break;
                        case "mistral": if (string.IsNullOrWhiteSpace(loaded.MistralApiKey)) { Console.WriteLine("!! Error: Mistral provider selected, but 'MistralApiKey' is missing."); isValid = false; } break;
                        case "groq": if (string.IsNullOrWhiteSpace(loaded.GroqApiKey)) { Console.WriteLine("!! Error: Groq provider selected, but 'GroqApiKey' is missing."); isValid = false; } break;
                    }
                }


                loaded.MaxContext = loaded.MaxContext <= 0 ? defaultSettings.MaxContext : loaded.MaxContext;
                loaded.MaxTokens = loaded.MaxTokens <= 0 ? defaultSettings.MaxTokens : loaded.MaxTokens;
                loaded.ResetIdleSeconds = loaded.ResetIdleSeconds <= 0 ? defaultSettings.ResetIdleSeconds : loaded.ResetIdleSeconds;

                if (!isValid)
                {
                    Console.WriteLine("Settings validation failed. Please correct settings.json and restart.");
                    return null;
                }

                Console.WriteLine("Settings loaded and validated successfully.");
                return loaded;
            }
            catch (JsonException jsonEx) { Console.WriteLine($"!! Error reading settings file {settingsFilePath} (JSON format issue): {jsonEx.Message}"); return null; }
            catch (IOException ioEx) { Console.WriteLine($"!! Error accessing settings file {settingsFilePath}: {ioEx.Message}"); return null; }
            catch (Exception e) { Console.WriteLine($"!! Unexpected error loading settings file: {e.ToString()}"); return null; }
        }
    }

    // --- Helper: Add Trailing Separator ---
    static string AddTrailingSeparator(string path)
    {
        if (string.IsNullOrEmpty(path)) return Path.DirectorySeparatorChar.ToString();
        path = path.Trim();
        if (!path.EndsWith(Path.DirectorySeparatorChar.ToString()) && !path.EndsWith(Path.AltDirectorySeparatorChar.ToString()))
        {
            return path + Path.DirectorySeparatorChar;
        }
        return path;
    }

    // --- Helper: Capitalize First Letter ---
    static string Capitalize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return text ?? string.Empty;
        if (text.Length == 1) return char.ToUpperInvariant(text[0]).ToString();
        // Capitalize first, lowercase rest for names
        return char.ToUpperInvariant(text[0]) + text.Substring(1).ToLowerInvariant();
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
    public string OpenAiApiKey { get; set; } = ""; // Used for OpenAI & potentially others if name is reused
    public string OpenAiModel { get; set; } = Models.ChatGpt3_5Turbo;

    // --- Ollama Specific ---
    public string OllamaBaseUrl { get; set; } = "http://localhost:11434";
    public string OllamaModel { get; set; } = "gemma3:4b";

    // --- Gemini Specific --- NEW ---
    public string GeminiApiKey { get; set; } = "";
    public string GeminiModel { get; set; } = "gemini-2.0-flash"; // Or "gemini-pro", etc.

    // --- Mistral Specific --- NEW ---
    public string MistralApiKey { get; set; } = "";
    public string MistralModel { get; set; } = "mistral-large-latest"; // Or "mistral-small", "open-mistral-7b" etc.
    // Optional: Mistral API URL if not using default
    // public string MistralApiUrl { get; set; } = "https://api.mistral.ai";

    // --- Groq Specific --- NEW ---
    public string GroqApiKey { get; set; } = "";
    public string GroqModel { get; set; } = "llama-3.1-8b-instant"; // Or "mixtral-8x7b-32768", "gemma-7b-it" etc.
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
