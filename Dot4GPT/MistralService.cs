using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization; // For attributes if needed
using System.Threading.Tasks;

public class MistralService : ILLMService
{
    private readonly HttpClient _httpClient;
    private readonly Settings _settings;
    private readonly string _mistralApiUrl = "https://api.mistral.ai"; // Default URL


    // --- Nested Classes for Mistral's JSON structure (OpenAI compatible) ---
    private class MistralMessage
    {
        [JsonPropertyName("role")]
        public string Role { get; set; } = "";
        [JsonPropertyName("content")]
        public string Content { get; set; } = "";
    }

    private class MistralChatRequest
    {
        [JsonPropertyName("model")]
        public string Model { get; set; } = "";
        [JsonPropertyName("messages")]
        public List<MistralMessage> Messages { get; set; } = new List<MistralMessage>();
        // Optional parameters matching OpenAI spec
        [JsonPropertyName("max_tokens")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] // Don't send if null
        public int? MaxTokens { get; set; }
        // public double? Temperature { get; set; } = 0.7;
        // public bool Stream { get; set; } = false; // Not streaming
    }

    private class MistralChatResponseChoice
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }
        [JsonPropertyName("message")]
        public MistralMessage? Message { get; set; }
        [JsonPropertyName("finish_reason")]
        public string? FinishReason { get; set; }
    }

    private class MistralChatResponse
    {
        [JsonPropertyName("id")]
        public string? Id { get; set; }
        [JsonPropertyName("object")]
        public string? Object { get; set; }
        [JsonPropertyName("created")]
        public long Created { get; set; }
        [JsonPropertyName("model")]
        public string? Model { get; set; }
        [JsonPropertyName("choices")]
        public List<MistralChatResponseChoice>? Choices { get; set; }
        // Usage object omitted for brevity
        [JsonPropertyName("error")]
        public MistralError? Error { get; set; } // Handle potential API errors
    }

    private class MistralError // Structure for error response
    {
        [JsonPropertyName("message")]
        public string? Message { get; set; }
        [JsonPropertyName("type")]
        public string? Type { get; set; }
        [JsonPropertyName("param")]
        public string? Param { get; set; }
        [JsonPropertyName("code")]
        public string? Code { get; set; }
    }
    // --- End Nested Classes ---


    public MistralService(Settings settings)
    {
        _settings = settings;
        if (string.IsNullOrWhiteSpace(settings.MistralApiKey))
        {
            throw new ArgumentNullException(nameof(settings.MistralApiKey), "Mistral API Key is missing in settings.");
        }
        if (string.IsNullOrWhiteSpace(settings.MistralModel))
        {
            throw new ArgumentNullException(nameof(settings.MistralModel), "Mistral Model is missing in settings.");
        }

        // Allow overriding API URL via settings if needed in the future
        // _mistralApiUrl = settings.MistralApiUrl ?? _mistralApiUrl;

        _httpClient = new HttpClient();
        _httpClient.BaseAddress = new Uri(_mistralApiUrl);
        // Set Authorization header with Bearer token
        _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", settings.MistralApiKey);
        _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    public async Task<string> GetChatCompletionAsync(List<GenericChatMessage> messages, Settings settings)
    {
        // 1. Convert messages to Mistral/OpenAI format
        var mistralMessages = messages.Select(m => new MistralMessage
        {
            Role = m.Role.ToString().ToLowerInvariant(), // system, user, assistant
            Content = m.Content
        }).ToList();

        // 2. Create request payload
        var requestPayload = new MistralChatRequest
        {
            Model = _settings.MistralModel,
            Messages = mistralMessages,
            MaxTokens = settings.MaxTokens > 0 ? settings.MaxTokens : null // Set max_tokens if > 0
        };

        // 3. Serialize payload
        string jsonPayload = "";
        // Use options to handle nulls correctly if needed
        var serializeOptions = new JsonSerializerOptions
        {
            // PropertyNamingPolicy = JsonNamingPolicy.CamelCase, // If property names differ in casing
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull // Important for optional parameters
        };
        try
        {
            jsonPayload = JsonSerializer.Serialize(requestPayload, serializeOptions);
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to serialize Mistral request payload: {ex.Message}");
        }

        // 4. Create HTTP content and make POST request
        using var httpContent = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
        HttpResponseMessage response;
        try
        {
            // Endpoint is typically "/v1/chat/completions"
            response = await _httpClient.PostAsync("/v1/chat/completions", httpContent);
        }
        catch (HttpRequestException ex)
        {
            throw new Exception($"Network error calling Mistral API at {_httpClient.BaseAddress}v1/chat/completions: {ex.Message}", ex);
        }
        catch (TaskCanceledException ex)
        {
            throw new Exception($"Mistral API call timed out: {ex.Message}", ex);
        }

        // 5. Read response body
        string responseBody = await response.Content.ReadAsStringAsync();

        // 6. Deserialize response (handling potential errors)
        MistralChatResponse? mistralResponse;
        try
        {
            var deserializeOptions = new JsonSerializerOptions { PropertyNameCaseInsensitive = true }; // Be flexible with casing
            mistralResponse = JsonSerializer.Deserialize<MistralChatResponse>(responseBody, deserializeOptions);

            // Check for error object within the JSON response
            if (mistralResponse?.Error != null)
            {
                throw new Exception($"Mistral API returned an error: {mistralResponse.Error.Message} (Type: {mistralResponse.Error.Type}, Code: {mistralResponse.Error.Code})");
            }

            // Also check HTTP status code AFTER trying to deserialize, as error might be in JSON body
            if (!response.IsSuccessStatusCode)
            {
                // If deserialization didn't find an error object, throw based on status code
                throw new Exception($"Mistral API request failed with status code {response.StatusCode}. Response body: {responseBody}");
            }
        }
        catch (JsonException ex)
        {
            // If deserialization fails, include response body for debugging
            throw new Exception($"Failed to deserialize Mistral response: {ex.Message}. Status Code: {response.StatusCode}. Response body: {responseBody}");
        }


        // 7. Extract assistant reply
        string? assistantReply = mistralResponse?.Choices?.FirstOrDefault()?
                                        .Message?.Content?.Trim();

        if (string.IsNullOrEmpty(assistantReply))
        {
            string finishReason = mistralResponse?.Choices?.FirstOrDefault()?.FinishReason ?? "Unknown";
            // Check if finish reason indicates a problem (e.g., length limit, content filter)
            if (finishReason.ToLowerInvariant() != "stop" && finishReason.ToLowerInvariant() != "eos") // "eos" used by some models
            {
                throw new Exception($"Mistral response was empty or incomplete. Finish Reason: {finishReason}");
            }
            // If reason was stop/eos but content is empty, treat as empty response.
            throw new Exception($"Mistral response content was empty. Finish Reason: {finishReason}.");
        }

        return assistantReply;
    }
}