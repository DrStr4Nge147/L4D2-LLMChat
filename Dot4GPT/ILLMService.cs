// Add this below the GenericChatMessage definition or in a new file (e.g., ILLMService.cs)

public interface ILLMService
{
    // Takes our generic messages and settings, returns the AI's response string
    Task<string> GetChatCompletionAsync(List<GenericChatMessage> messages, Settings settings);
}