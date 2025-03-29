using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

public class GeminiService : ILLMService
{
    private readonly HttpClient _httpClient;
    private readonly Settings _settings;
    private readonly string _geminiApiUrl;

    // --- Nested Classes for Gemini REST API JSON structure ---
    private class GeminiPart
    {
        [JsonPropertyName("text")]
        public string Text { get; set; } = "";
    }

    private class GeminiContent
    {
        [JsonPropertyName("role")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] // Role is omitted for system instructions
        public string? Role { get; set; } // "user" or "model"

        [JsonPropertyName("parts")]
        public List<GeminiPart> Parts { get; set; } = new List<GeminiPart>();
    }

    private class GeminiSafetySetting
    {
        [JsonPropertyName("category")]
        public string Category { get; set; } = ""; // e.g., HARM_CATEGORY_SEXUALLY_EXPLICIT

        [JsonPropertyName("threshold")]
        public string Threshold { get; set; } = ""; // e.g., BLOCK_MEDIUM_AND_ABOVE
    }

    private class GeminiGenerationConfig
    {
        // [JsonPropertyName("temperature")]
        // public double? Temperature { get; set; } = 0.7;

        [JsonPropertyName("maxOutputTokens")]
        public int? MaxOutputTokens { get; set; }

        // Add other config like stopSequences, topP, topK if needed
    }

    private class GeminiChatRequest
    {
        // System instruction is part of the top-level request in v1beta
        [JsonPropertyName("system_instruction")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public GeminiContent? SystemInstruction { get; set; }

        [JsonPropertyName("contents")]
        public List<GeminiContent> Contents { get; set; } = new List<GeminiContent>();

        // Safety settings are often optional, defaults are usually reasonable
        // [JsonPropertyName("safetySettings")]
        // public List<GeminiSafetySetting> SafetySettings { get; set; }

        [JsonPropertyName("generationConfig")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public GeminiGenerationConfig? GenerationConfig { get; set; }
    }

    // --- Response Classes ---
    private class GeminiCandidate
    {
        [JsonPropertyName("content")]
        public GeminiContent? Content { get; set; }

        [JsonPropertyName("finishReason")]
        public string? FinishReason { get; set; }
        // Index, SafetyRatings etc. also available
    }

    private class GeminiPromptFeedback // In case the prompt itself is blocked
    {
        [JsonPropertyName("blockReason")]
        public string? BlockReason { get; set; }
        // SafetyRatings also available
    }


    private class GeminiChatResponse
    {
        [JsonPropertyName("candidates")]
        public List<GeminiCandidate>? Candidates { get; set; }

        [JsonPropertyName("promptFeedback")]
        public GeminiPromptFeedback? PromptFeedback { get; set; }

        // If the API itself returns an error object
        [JsonPropertyName("error")]
        public GeminiError? Error { get; set; }
    }

    private class GeminiError // Structure for error response (if API returns one in JSON)
    {
        [JsonPropertyName("code")]
        public int Code { get; set; }
        [JsonPropertyName("message")]
        public string? Message { get; set; }
        [JsonPropertyName("status")]
        public string? Status { get; set; }
    }
    // --- End Nested Classes ---


    public GeminiService(Settings settings)
    {
        _settings = settings;
        if (string.IsNullOrWhiteSpace(settings.GeminiApiKey))
        {
            throw new ArgumentNullException(nameof(settings.GeminiApiKey), "Gemini API Key is missing in settings.");
        }
        if (string.IsNullOrWhiteSpace(settings.GeminiModel))
        {
            throw new ArgumentNullException(nameof(settings.GeminiModel), "Gemini Model is missing in settings.");
        }

        _httpClient = new HttpClient();
        // Base URL for the Gemini REST API
        _geminiApiUrl = $"https://generativelanguage.googleapis.com/v1beta/models/{_settings.GeminiModel}:generateContent?key={_settings.GeminiApiKey}";
        _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    public async Task<string> GetChatCompletionAsync(List<GenericChatMessage> messages, Settings settings)
    {
        // 1. Separate System Instruction & Convert messages to Gemini format
        GeminiContent? systemInstruction = null;
        var geminiContents = new List<GeminiContent>();

        foreach (var msg in messages)
        {
            string role = msg.Role switch
            {
                ChatRole.User => "user",
                ChatRole.Assistant => "model",
                ChatRole.System => "system", // Temporary role
                _ => "user"
            };

            var contentPart = new GeminiPart { Text = msg.Content };

            if (role == "system")
            {
                // System prompt goes into a specific field in the request body
                systemInstruction = new GeminiContent { Parts = { contentPart } };
                // DO NOT add system message to the main 'contents' list for Gemini REST API
            }
            else
            {
                geminiContents.Add(new GeminiContent { Role = role, Parts = { contentPart } });
            }
        }

        // 2. Create request payload
        var requestPayload = new GeminiChatRequest
        {
            SystemInstruction = systemInstruction, // Add system instruction here
            Contents = geminiContents,
            GenerationConfig = new GeminiGenerationConfig
            {
                MaxOutputTokens = settings.MaxTokens > 0 ? settings.MaxTokens : null
            }
            // Add safety settings if needed
        };

        // 3. Serialize payload
        string jsonPayload = "";
        var serializeOptions = new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };
        try
        {
            jsonPayload = JsonSerializer.Serialize(requestPayload, serializeOptions);
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to serialize Gemini request payload: {ex.Message}");
        }


        // 4. Create HTTP content and make POST request
        using var httpContent = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
        HttpResponseMessage response;
        try
        {
            // URL already includes model and API key
            response = await _httpClient.PostAsync(_geminiApiUrl, httpContent);
        }
        catch (HttpRequestException ex)
        {
            throw new Exception($"Network error calling Gemini API: {ex.Message}", ex);
        }
        catch (TaskCanceledException ex)
        {
            throw new Exception($"Gemini API call timed out: {ex.Message}", ex);
        }

        // 5. Read response body
        string responseBody = await response.Content.ReadAsStringAsync();

        // 6. Deserialize response (check for HTTP errors AND API errors in JSON)
        GeminiChatResponse? geminiResponse;
        try
        {
            var deserializeOptions = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            geminiResponse = JsonSerializer.Deserialize<GeminiChatResponse>(responseBody, deserializeOptions);

            // Check for error object within the JSON response first
            if (geminiResponse?.Error != null)
            {
                throw new Exception($"Gemini API returned an error: {geminiResponse.Error.Message} (Status: {geminiResponse.Error.Status}, Code: {geminiResponse.Error.Code})");
            }

            // Then check HTTP status code if no JSON error was found
            if (!response.IsSuccessStatusCode)
            {
                throw new Exception($"Gemini API request failed with status code {response.StatusCode}. Response body: {responseBody}");
            }
        }
        catch (JsonException ex)
        {
            // Include response body if deserialization fails
            throw new Exception($"Failed to deserialize Gemini response: {ex.Message}. Status Code: {response.StatusCode}. Response body: {responseBody}");
        }

        // 7. Check for blocked prompt feedback
        if (geminiResponse?.Candidates == null || !geminiResponse.Candidates.Any())
        {
            string feedback = geminiResponse?.PromptFeedback?.BlockReason != null
               ? $"Prompt Block Reason: {geminiResponse.PromptFeedback.BlockReason}"
               : "No candidates returned and no specific prompt feedback.";
            throw new Exception($"Gemini returned no candidates. {feedback}");
        }

        // 8. Extract assistant reply
        string? assistantReply = geminiResponse?.Candidates?.FirstOrDefault()?
                                        .Content?.Parts?.FirstOrDefault()?
                                        .Text?.Trim();

        if (string.IsNullOrEmpty(assistantReply))
        {
            string finishReason = geminiResponse?.Candidates?.FirstOrDefault()?.FinishReason ?? "Unknown";
            if (finishReason.ToLowerInvariant() != "stop")
            {
                throw new Exception($"Gemini response was empty or incomplete. Finish Reason: {finishReason}");
            }
            throw new Exception($"Gemini response content was empty. Finish Reason: {finishReason}.");
        }

        return assistantReply;
    }
}