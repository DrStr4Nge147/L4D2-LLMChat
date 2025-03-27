using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json; // Using System.Text.Json
using System.Threading.Tasks;

public class OllamaService : ILLMService // Implement our interface
{
    private readonly HttpClient _httpClient;
    private readonly Settings _settings;

    // --- Nested Classes for Ollama's JSON structure ---
    private class OllamaMessage
    {
        public string role { get; set; } = "";
        public string content { get; set; } = "";
    }

    private class OllamaChatRequest
    {
        public string model { get; set; } = "";
        public List<OllamaMessage> messages { get; set; } = new List<OllamaMessage>();
        public bool stream { get; set; } = false; // We want the full response, not streaming
        public OllamaOptions? options { get; set; } // Optional parameters
    }

    private class OllamaOptions // Example options - consult Ollama API docs
    {
        // public int num_predict { get; set; } // Equivalent to max_tokens, but Ollama's default might be okay
        // public double temperature { get; set; } = 0.7;
        // Add other options if needed: top_p, top_k, stop sequences etc.
    }

    private class OllamaChatResponse
    {
        public string model { get; set; } = "";
        public DateTime created_at { get; set; }
        public OllamaMessage? message { get; set; } // The assistant's reply message
        public bool done { get; set; }
        // Other fields like timings, eval_count etc. are also present but often not needed here
        public string? error { get; set; } // Ollama might return an error message here
    }
    // --- End Nested Classes ---


    // Constructor: Takes settings, creates HttpClient
    public OllamaService(Settings settings)
    {
        _settings = settings;

        // Validate required Ollama settings
        if (string.IsNullOrWhiteSpace(settings.OllamaBaseUrl))
        {
            throw new ArgumentNullException(nameof(settings.OllamaBaseUrl), "Ollama Base URL is missing in settings.");
        }
        if (string.IsNullOrWhiteSpace(settings.OllamaModel))
        {
            throw new ArgumentNullException(nameof(settings.OllamaModel), "Ollama Model is missing in settings.");
        }

        _httpClient = new HttpClient();
        // Ensure BaseAddress ends with a slash
        string baseUrl = settings.OllamaBaseUrl.TrimEnd('/') + "/";
        _httpClient.BaseAddress = new Uri(baseUrl);
        _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    // The core implementation of the interface method
    public async Task<string> GetChatCompletionAsync(List<GenericChatMessage> messages, Settings settings)
    {
        // 1. Convert our GenericChatMessage list to Ollama's format
        var ollamaMessages = messages.Select(m => new OllamaMessage
        {
            // Map roles: System.Text.Json is case-sensitive by default, Ollama expects lowercase
            role = m.Role.ToString().ToLowerInvariant(),
            content = m.Content
        }).ToList();

        // 2. Create the request payload
        var requestPayload = new OllamaChatRequest
        {
            model = _settings.OllamaModel, // Use model from settings
            messages = ollamaMessages,
            stream = false, // Important: Get response at once
            // options = new OllamaOptions { num_predict = settings.MaxTokens } // Control max reply length if desired
        };

        // 3. Serialize the request payload to JSON
        string jsonPayload = "";
        try
        {
            // Use System.Text.Json options for potential case insensitivity if needed later
            jsonPayload = JsonSerializer.Serialize(requestPayload/*, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }*/);
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to serialize Ollama request payload: {ex.Message}");
        }

        // 4. Create the HTTP request content
        using var httpContent = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

        // 5. Make the POST request to Ollama's chat endpoint
        HttpResponseMessage response;
        try
        {
            // The endpoint is typically relative like "api/chat"
            response = await _httpClient.PostAsync("api/chat", httpContent);
        }
        catch (HttpRequestException ex)
        {
            // Network errors, Ollama server down, DNS issues etc.
            throw new Exception($"Network error calling Ollama API at {_httpClient.BaseAddress}api/chat: {ex.Message}", ex);
        }
        catch (TaskCanceledException ex) // Handle timeouts
        {
            throw new Exception($"Ollama API call timed out: {ex.Message}", ex);
        }


        // 6. Read the response content
        string responseBody = await response.Content.ReadAsStringAsync();

        // 7. Check for HTTP errors first
        if (!response.IsSuccessStatusCode)
        {
            // Try to include Ollama's error message if available in the response body
            string errorDetail = responseBody; // Use the body as detail if possible
            if (string.IsNullOrWhiteSpace(errorDetail)) errorDetail = $"Status Code: {response.StatusCode}";
            throw new Exception($"Ollama API request failed: {errorDetail}");
        }

        // 8. Deserialize the successful JSON response
        OllamaChatResponse? ollamaResponse;
        try
        {
            ollamaResponse = JsonSerializer.Deserialize<OllamaChatResponse>(responseBody/*, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }*/);
        }
        catch (JsonException ex)
        {
            throw new Exception($"Failed to deserialize Ollama response: {ex.Message}. Response body: {responseBody}");
        }

        // 9. Check for errors within the Ollama JSON response itself
        if (ollamaResponse?.error != null)
        {
            throw new Exception($"Ollama returned an error: {ollamaResponse.error}");
        }

        // 10. Extract the assistant's message content
        string? assistantReply = ollamaResponse?.message?.content?.Trim();

        if (string.IsNullOrEmpty(assistantReply))
        {
            // Handle cases where the response format is unexpected or content is empty
            // Consider throwing an exception or returning a specific error message
            throw new Exception("Ollama response received, but assistant message content was missing or empty.");
            // return "[Ollama response was empty]"; // Alternative: return error string
        }

        return assistantReply;
    }

    // Optional: Implement IDisposable if HttpClient is managed per instance
    // public void Dispose()
    // {
    //     _httpClient?.Dispose();
    // }
}