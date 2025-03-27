public enum ChatRole
{
    System,
    User,
    Assistant
}

public class GenericChatMessage
{
    public ChatRole Role { get; set; }
    public string Content { get; set; }

    public GenericChatMessage(ChatRole role, string content)
    {
        Role = role;
        Content = content;
    }

    // Helper methods (optional but convenient)
    public static GenericChatMessage FromSystem(string content) => new GenericChatMessage(ChatRole.System, content);
    public static GenericChatMessage FromUser(string content) => new GenericChatMessage(ChatRole.User, content);
    public static GenericChatMessage FromAssistant(string content) => new GenericChatMessage(ChatRole.Assistant, content);
}