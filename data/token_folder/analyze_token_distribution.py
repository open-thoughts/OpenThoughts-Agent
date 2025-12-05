#!/usr/bin/env python3
"""
Analyze token distribution in the DCAgent freelancer dataset using Qwen 3 tokenizer.
"""
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm

print("Loading dataset...")
dataset = load_dataset('DCAgent/taskmaster2-10k-traces', split='train')

print("Loading Qwen 3 tokenizer...")
# Load Qwen 3 tokenizer (Qwen2.5 is the latest version)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

print("Processing conversations and tokenizing...")
tokens_per_conversation = []
tokens_per_episode = []  # Each episode is a user-assistant turn pair
tokens_per_message = []
episodes_per_trace = []  # Number of episodes in each conversation/trace

for entry in tqdm(dataset, desc="Tokenizing"):
    conversations = entry['conversations']

    # Calculate tokens for this conversation (sum of all messages)
    conversation_tokens = 0
    episode_count = 0

    # Find user-assistant pairs (episodes)
    i = 0
    while i < len(conversations):
        message = conversations[i]
        content = message['content']
        role = message['role']

        tokens = tokenizer.encode(content, add_special_tokens=False)
        message_tokens = len(tokens)
        conversation_tokens += message_tokens
        tokens_per_message.append(message_tokens)

        # If this is a user message, look for the next assistant message to form an episode
        if role == 'user' and i + 1 < len(conversations):
            next_message = conversations[i + 1]
            if next_message['role'] == 'assistant':
                # This is a user-assistant pair (episode)
                next_content = next_message['content']
                next_tokens = tokenizer.encode(next_content, add_special_tokens=False)
                next_message_tokens = len(next_tokens)

                episode_tokens = message_tokens + next_message_tokens
                tokens_per_episode.append(episode_tokens)
                episode_count += 1

                # Add next message tokens to conversation total and skip it
                conversation_tokens += next_message_tokens
                tokens_per_message.append(next_message_tokens)
                i += 2
                continue

        i += 1

    tokens_per_conversation.append(conversation_tokens)
    episodes_per_trace.append(episode_count)

print(f"\nStatistics:")
print(f"Total conversations: {len(tokens_per_conversation)}")
print(f"Total episodes: {len(tokens_per_episode)}")
print(f"Total messages: {len(tokens_per_message)}")

print(f"\nTokens per conversation:")
print(f"  Mean: {np.mean(tokens_per_conversation):.2f}")
print(f"  Median: {np.median(tokens_per_conversation):.2f}")
print(f"  Min: {np.min(tokens_per_conversation)}")
print(f"  Max: {np.max(tokens_per_conversation)}")
print(f"  Std: {np.std(tokens_per_conversation):.2f}")

print(f"\nTokens per episode:")
print(f"  Mean: {np.mean(tokens_per_episode):.2f}")
print(f"  Median: {np.median(tokens_per_episode):.2f}")
print(f"  Min: {np.min(tokens_per_episode)}")
print(f"  Max: {np.max(tokens_per_episode)}")
print(f"  Std: {np.std(tokens_per_episode):.2f}")

print(f"\nEpisodes per trace:")
print(f"  Mean: {np.mean(episodes_per_trace):.2f}")
print(f"  Median: {np.median(episodes_per_trace):.2f}")
print(f"  Min: {np.min(episodes_per_trace)}")
print(f"  Max: {np.max(episodes_per_trace)}")
print(f"  Std: {np.std(episodes_per_trace):.2f}")

# Create histograms - 3 panel figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Histogram 1: Tokens per conversation
ax1.hist(tokens_per_conversation, bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of Tokens', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Tokens per Conversation', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axvline(np.mean(tokens_per_conversation), color='red', linestyle='--',
            label=f'Mean: {np.mean(tokens_per_conversation):.0f}', linewidth=2)
ax1.axvline(np.median(tokens_per_conversation), color='green', linestyle='--',
            label=f'Median: {np.median(tokens_per_conversation):.0f}', linewidth=2)
ax1.legend()

# Histogram 2: Tokens per episode
ax2.hist(tokens_per_episode, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax2.set_xlabel('Number of Tokens', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Tokens per Episode', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.axvline(np.mean(tokens_per_episode), color='red', linestyle='--',
            label=f'Mean: {np.mean(tokens_per_episode):.0f}', linewidth=2)
ax2.axvline(np.median(tokens_per_episode), color='green', linestyle='--',
            label=f'Median: {np.median(tokens_per_episode):.0f}', linewidth=2)
ax2.legend()

# Histogram 3: Episodes per trace
ax3.hist(episodes_per_trace, bins=30, edgecolor='black', alpha=0.7, color='green')
ax3.set_xlabel('Number of Episodes', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of Episodes per Trace', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.axvline(np.mean(episodes_per_trace), color='red', linestyle='--',
            label=f'Mean: {np.mean(episodes_per_trace):.2f}', linewidth=2)
ax3.axvline(np.median(episodes_per_trace), color='darkgreen', linestyle='--',
            label=f'Median: {np.median(episodes_per_trace):.0f}', linewidth=2)
ax3.legend()

plt.tight_layout()
plt.savefig('token_distribution_histograms.png', dpi=300, bbox_inches='tight')
print("\nHistograms saved to 'token_distribution_histograms.png'")

# Also create separate plots for better visibility
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(tokens_per_conversation, bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of Tokens', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Tokens per Conversation', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axvline(np.mean(tokens_per_conversation), color='red', linestyle='--',
            label=f'Mean: {np.mean(tokens_per_conversation):.0f}', linewidth=2)
ax1.axvline(np.median(tokens_per_conversation), color='green', linestyle='--',
            label=f'Median: {np.median(tokens_per_conversation):.0f}', linewidth=2)
ax1.legend()
plt.tight_layout()
plt.savefig('tokens_per_conversation.png', dpi=300, bbox_inches='tight')
print("Conversation histogram saved to 'tokens_per_conversation.png'")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.hist(tokens_per_episode, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax2.set_xlabel('Number of Tokens', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Tokens per Episode', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.axvline(np.mean(tokens_per_episode), color='red', linestyle='--',
            label=f'Mean: {np.mean(tokens_per_episode):.0f}', linewidth=2)
ax2.axvline(np.median(tokens_per_episode), color='green', linestyle='--',
            label=f'Median: {np.median(tokens_per_episode):.0f}', linewidth=2)
ax2.legend()
plt.tight_layout()
plt.savefig('tokens_per_episode.png', dpi=300, bbox_inches='tight')
print("Episode histogram saved to 'tokens_per_episode.png'")

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.hist(episodes_per_trace, bins=30, edgecolor='black', alpha=0.7, color='green')
ax3.set_xlabel('Number of Episodes', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of Episodes per Trace', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.axvline(np.mean(episodes_per_trace), color='red', linestyle='--',
            label=f'Mean: {np.mean(episodes_per_trace):.2f}', linewidth=2)
ax3.axvline(np.median(episodes_per_trace), color='darkgreen', linestyle='--',
            label=f'Median: {np.median(episodes_per_trace):.0f}', linewidth=2)
ax3.legend()
plt.tight_layout()
plt.savefig('episodes_per_trace.png', dpi=300, bbox_inches='tight')
print("Episodes per trace histogram saved to 'episodes_per_trace.png'")

plt.close('all')
print("\nAnalysis complete!")
