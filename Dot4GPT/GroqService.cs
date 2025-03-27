using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

// NOTE: This class is nearly identical to MistralService due to OpenAI API compatibility.
// Consider creating a shared base class in the future if adding more OpenAI-like APIs.
public class GroqService : ILLMService
{
    private readonly HttpClient _httpClient;
    private readonly Settings _settings;
    private readonly string _groqApiUrl = "https://api.groq.com/openai"; // Default URL

    // --- Nested Classes (Identical to Mistral's OpenAI-compatible ones) ---
    private class GroqMessage
    {
        [JsonPropertyName("role")]
        public string Role { get; set; } = "";
        [JsonPropertyName("content")]
        public string Content { get; set; } = "";
    }

    private class GroqChatRequest
    {
        [JsonPropertyName("model")]
        public string Model { get; set; } = "";
        [JsonPropertyName("messages")]
        public List<GroqMessage> Messages { get; set; } = new List<GroqMessage>();
        [JsonPropertyName("max_tokens")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public int? MaxTokens { get; set; }
        // public double? Temperature { get; set; } = 0.7;
        // public bool Stream { get; set; } = false;
    }

    private class GroqChatResponseChoice
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }
        [JsonPropertyName("message")]
        public GroqMessage? Message { get; set; }
        [JsonPropertyName("finish_reason")]
        public string? FinishReason { get; set; }
    }

    private class GroqChatResponse
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
        public List<GroqChatResponseChoice>? Choices { get; set; }
        // Groq might have a different error structure, adjust if needed
        // For now, assume simple HTTP errors or OpenAI-like error object
        // Add error property if Groq's specific error response is known
        // [JsonPropertyName("error")]
        // public GroqError? Error { get; set; }
    }
    // Define GroqError if needed
    // --- End Nested Classes ---

    public GroqService(Settings settings)
    {
        _settings = settings;
        if (string.IsNullOrWhiteSpace(settings.GroqApiKey))
        {
            throw new ArgumentNullException(nameof(settings.GroqApiKey), "Groq API Key is missing in settings.");
        }
        if (string.IsNullOrWhiteSpace(settings.GroqModel))
        {
            throw new ArgumentNullException(nameof(settings.GroqModel), "Groq Model is missing in settings.");
        }

        // Allow overriding API URL via settings if needed
        // _groqApiUrl = settings.GroqApiUrl ?? _groqApiUrl;

        _httpClient = new HttpClient();
        // Groq API endpoint needs /v1 suffix
        _httpClient.BaseAddress = new Uri(_groqApiUrl.TrimEnd('/') + "/"); // Ensure trailing slash
        _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", settings.GroqApiKey);
        _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    public async Task<string> GetChatCompletionAsync(List<GenericChatMessage> messages, Settings settings)
    {
        // 1. Convert messages
        var groqMessages = messages.Select(m => new GroqMessage
        {
            Role = m.Role.ToString().ToLowerInvariant(),
            Content = m.Content
        }).ToList();

        // 2. Create request payload
        var requestPayload = new GroqChatRequest
        {
            Model = _settings.GroqModel,
            Messages = groqMessages,
            MaxTokens = settings.MaxTokens > 0 ? settings.MaxTokens : null
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
            throw new Exception($"Failed to serialize Groq request payload: {ex.Message}");
        }

        // 4. Create HTTP content and make POST request
        using var httpContent = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
        HttpResponseMessage response;
        try
        {
            // Endpoint is typically "v1/chat/completions" relative to base
            response = await _httpClient.PostAsync("v1/chat/completions", httpContent);
        }
        catch (HttpRequestException ex)
        {
            throw new Exception($"Network error calling Groq API at {_httpClient.BaseAddress}v1/chat/completions: {ex.Message}", ex);
        }
        catch (TaskCanceledException ex)
        {
            throw new Exception($"Groq API call timed out: {ex.Message}", ex);
        }

        // 5. Read response body
        string responseBody = await response.Content.ReadAsStringAsync();

        // 6. Deserialize response (handling potential errors)
        GroqChatResponse? groqResponse;
        try
        {
            var deserializeOptions = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            groqResponse = JsonSerializer.Deserialize<GroqChatResponse>(responseBody, deserializeOptions);

            // Check for HTTP errors first - Groq might just return non-200 on failure
            if (!response.IsSuccessStatusCode)
            {
                // Try to parse standard OpenAI error format if present
                // You might need a specific GroqError class if their errors differ
                try
                {
                    var errorResponse = JsonSerializer.Deserialize<OpenAIErrorResponse>(responseBody, deserializeOptions);
                    if (errorResponse?.Error != null)
                    {
                        throw new Exception($"Groq API returned an error: {errorResponse.Error.Message} (Type: {errorResponse.Error.Type}, Code: {errorResponse.Error.Code})");
                    }
                }
                catch (JsonException) { /* Ignore if response isn't the error format */ }

                // Fallback generic error
                throw new Exception($"Groq API request failed with status code {response.StatusCode}. Response body: {responseBody}");
            }
        }
        catch (JsonException ex)
        {
            throw new Exception($"Failed to deserialize Groq response: {ex.Message}. Status Code: {response.StatusCode}. Response body: {responseBody}");
        }

        // 7. Extract assistant reply
        string? assistantReply = groqResponse?.Choices?.FirstOrDefault()?
                                        .Message?.Content?.Trim();

        if (string.IsNullOrEmpty(assistantReply))
        {
            string finishReason = groqResponse?.Choices?.FirstOrDefault()?.FinishReason ?? "Unknown";
            if (finishReason.ToLowerInvariant() != "stop")
            {
                throw new Exception($"Groq response was empty or incomplete. Finish Reason: {finishReason}");
            }
            throw new Exception($"Groq response content was empty. Finish Reason: {finishReason}.");
        }

        return assistantReply;
    }

    // Helper class for standard OpenAI error format (Groq might use this)
    private class OpenAIError
    {
        [JsonPropertyName("message")] public string? Message { get; set; }
        [JsonPropertyName("type")] public string? Type { get; set; }
        [JsonPropertyName("param")] public string? Param { get; set; }
        [JsonPropertyName("code")] public string? Code { get; set; }
    }
    private class OpenAIErrorResponse
    {
        [JsonPropertyName("error")] public OpenAIError? Error { get; set; }
    }
}