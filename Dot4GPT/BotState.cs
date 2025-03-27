using System;
using System.Collections.Generic;

// Make sure this class is public if Program.cs is in a different namespace (unlikely here)
public class BotState
{
    // Holds the message history (including system prompt) for this specific bot
    public List<GenericChatMessage> Messages { get; set; } = new List<GenericChatMessage>();

    // Tracks the last time a user message was received for this bot
    public DateTime LastInteraction { get; set; } = DateTime.Now;

    // --- ADD THIS LINE ---
    // Flag to prevent concurrent processing for the same bot
    public volatile bool IsProcessing = false;
    // --- END OF ADDED LINE ---

    // Optional: Constructor if needed, though default is fine here
    // public BotState() { }
}