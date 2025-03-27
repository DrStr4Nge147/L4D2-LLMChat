// Add this class, preferably in a new file (e.g., OpenAIServiceWrapper.cs)
using OpenAI.GPT3;
using OpenAI.GPT3.Managers;
using OpenAI.GPT3.ObjectModels.RequestModels;
using OpenAI.GPT3.ObjectModels; // For Models class
using System.Collections.Generic; // For List
using System.Linq; // For First()
using System.Threading.Tasks; // For Task

public class OpenAIServiceWrapper : ILLMService
{
    private readonly OpenAIService _openAiService;

    // Constructor takes the settings to get the API key
    public OpenAIServiceWrapper(Settings settings)
    {
        if (string.IsNullOrEmpty(settings.ApiKey))
        {
            throw new ArgumentNullException(nameof(settings.ApiKey), "OpenAI API Key is missing in settings.");
        }
        _openAiService = new OpenAIService(new OpenAiOptions()
        {
            ApiKey = settings.ApiKey
        });
    }

    public async Task<string> GetChatCompletionAsync(List<GenericChatMessage> messages, Settings settings)
    {
        // 1. Convert our GenericChatMessage list to the OpenAI-specific ChatMessage list
        var openAiMessages = messages.Select(m => new ChatMessage(m.Role.ToString().ToLower(), m.Content)).ToList();

        // 2. Create the request
        var completionResult = await _openAiService.ChatCompletion.CreateCompletion(new ChatCompletionCreateRequest
        {
            Messages = openAiMessages,
            Model = settings.OpenAiModel, // Use the model from settings
            MaxTokens = settings.MaxTokens
        });

        // 3. Process the response
        if (completionResult.Successful)
        {
            return completionResult.Choices.First().Message.Content;
        }
        else
        {
            string errorMsg = "OpenAI API call failed.";
            if (settings.APIErrors)
            {
                if (completionResult.Error?.Message != null)
                    errorMsg = $"OpenAI Error: {completionResult.Error.Message}";
                else if (completionResult.Error?.Code != null)
                    errorMsg = $"OpenAI Error Code: {completionResult.Error.Code}";
                else if (completionResult.Error != null)
                    errorMsg = $"OpenAI Error: {completionResult.Error}"; // Fallback
            }
            // We should throw an exception here to signal failure clearly
            // The main loop can then catch it and decide on the default message.
            throw new Exception(errorMsg);
        }
    }
}