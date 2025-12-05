#!/usr/bin/env python3

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import glob
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi

def extract_run_metadata(run_folder: str) -> Dict[str, Any]:
    """Extract metadata from run_metadata.json file."""
    metadata_path = os.path.join(run_folder, "run_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def find_all_task_folders(run_folder: str) -> List[str]:
    """Find all task folders in the run directory."""
    task_folders = []
    for item in os.listdir(run_folder):
        item_path = os.path.join(run_folder, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item not in ['tb.lock']:
            # Look for trial folders within task folders
            for trial_item in os.listdir(item_path):
                trial_path = os.path.join(item_path, trial_item)
                if os.path.isdir(trial_path) and trial_item.endswith(os.path.basename(run_folder)):
                    task_folders.append(trial_path)
    return sorted(task_folders)

def find_episode_folders(task_folder: str) -> List[str]:
    """Find all episode folders within a task folder."""
    agent_logs_path = os.path.join(task_folder, "agent-logs")
    if not os.path.exists(agent_logs_path):
        return []
    
    episode_folders = []
    for item in os.listdir(agent_logs_path):
        if item.startswith("episode-") and os.path.isdir(os.path.join(agent_logs_path, item)):
            episode_folders.append(os.path.join(agent_logs_path, item))
    
    return sorted(episode_folders)

def extract_conversation_from_episode(episode_folder: str, run_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract conversation data from a single episode folder."""
    debug_file = os.path.join(episode_folder, "debug.json")
    response_file = os.path.join(episode_folder, "response.json")
    
    if not os.path.exists(debug_file) or not os.path.exists(response_file):
        return None
    
    try:
        # Read debug.json for input messages
        with open(debug_file, 'r') as f:
            debug_data = json.load(f)
        
        # Read response.json for assistant response
        with open(response_file, 'r') as f:
            response_data = json.load(f)
        
        # Extract basic info
        conversation_data = {
            'conversations': [],
            'agent': 'terminus',  # Based on run metadata
            'model': run_metadata.get('model_name', 'unknown'),
            'date': run_metadata.get('start_time', ''),
            'task': None,
            'episode': os.path.basename(episode_folder),
            'run_id': run_metadata.get('run_id'),
            'trial_name': None
        }
        
        # Extract task name from folder path
        # Example path: .../blind-maze-explorer-5x5/blind-maze-explorer-5x5.1-of-1.2025-07-24__09-58-13/agent-logs/episode-0
        path_parts = episode_folder.split(os.sep)
        
        # Find the trial folder (contains .1-of-1. and ends with run_id)
        trial_folder = None
        run_id = run_metadata.get('run_id', '')
        for part in path_parts:
            if '.1-of-1.' in part and part.endswith(run_id):
                trial_folder = part
                break
        
        if trial_folder:
            conversation_data['trial_name'] = trial_folder
            # Extract base task name (remove .1-of-1.timestamp suffix)
            # Example: "blind-maze-explorer-5x5.1-of-1.2025-07-24__09-58-13" -> "blind-maze-explorer-5x5"
            if trial_folder.endswith('.' + run_id):
                base_name = trial_folder[:-len('.' + run_id)]
                # Remove the .1-of-1 suffix if present
                if base_name.endswith('.1-of-1'):
                    base_name = base_name[:-len('.1-of-1')]
                conversation_data['task'] = base_name
            else:
                conversation_data['task'] = trial_folder.split('.')[0]
        
        # Extract input messages from debug.json
        if 'input' in debug_data and isinstance(debug_data['input'], list):
            for msg in debug_data['input']:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Handle both string content and content with parts
                    content = msg['content']
                    if isinstance(content, list) and len(content) > 0:
                        if isinstance(content[0], dict) and 'text' in content[0]:
                            content = content[0]['text']
                    
                    conversation_data['conversations'].append({
                        'role': msg['role'],
                        'content': str(content)
                    })
        
        # Extract assistant response from response.json
        if response_data:
            # Convert the structured response to a string representation
            assistant_content = json.dumps(response_data, indent=2)
            conversation_data['conversations'].append({
                'role': 'assistant',
                'content': assistant_content
            })
        
        return conversation_data
        
    except Exception as e:
        print(f"Error processing episode {episode_folder}: {e}")
        return None

def process_run_folder(run_folder: str) -> List[Dict[str, Any]]:
    """Process an entire run folder and extract all conversation data."""
    print(f"Processing run folder: {run_folder}")
    
    # Extract run metadata
    run_metadata = extract_run_metadata(run_folder)
    print(f"Run ID: {run_metadata.get('run_id')}")
    print(f"Model: {run_metadata.get('model_name')}")
    print(f"Tasks: {len(run_metadata.get('task_ids', []))}")
    
    # Find all task folders
    task_folders = find_all_task_folders(run_folder)
    print(f"Found {len(task_folders)} task trial folders")
    
    all_conversations = []
    
    for task_folder in task_folders:
        print(f"\nProcessing task folder: {os.path.basename(task_folder)}")
        
        # Find episode folders
        episode_folders = find_episode_folders(task_folder)
        print(f"  Found {len(episode_folders)} episodes")
        
        # Process each episode
        for episode_folder in episode_folders:
            conversation_data = extract_conversation_from_episode(episode_folder, run_metadata)
            if conversation_data and conversation_data['conversations']:
                all_conversations.append(conversation_data)
    
    print(f"\nExtracted {len(all_conversations)} conversations total")
    return all_conversations

def upload_to_huggingface(dataset_rows: List[Dict], repo_id: str = "mlfoundations-dev/terminal-bench-traces-local"):
    """Upload dataset to Hugging Face repository."""
    if not dataset_rows:
        print("No data to upload.")
        return False
    
    # Check for HF token
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if not hf_token:
        print("✗ No Hugging Face token found. Please set HUGGINGFACE_TOKEN or HF_TOKEN environment variable.")
        print("  You can get a token from https://huggingface.co/settings/tokens")
        return False
    
    print(f"\nPreparing to upload {len(dataset_rows)} conversations to {repo_id}")
    
    # Create dataset
    dataset = Dataset.from_list(dataset_rows)
    
    print("Dataset created with columns:", dataset.column_names)
    print("Sample data:")
    if len(dataset) > 0:
        sample = dataset[0]
        for key, value in sample.items():
            if key == 'conversations' and isinstance(value, list):
                print(f"  {key}: [{len(value)} messages]")
                if len(value) > 0:
                    print(f"    First message preview: {str(value[0])[:100]}...")
            else:
                print(f"  {key}: {str(value)[:100]}")
    
    # Upload to Hugging Face
    try:
        dataset.push_to_hub(repo_id, token=hf_token)
        print(f"✓ Successfully uploaded dataset to {repo_id}")
        return True
    except Exception as e:
        print(f"✗ Error uploading to Hugging Face: {e}")
        return False

def save_local_results(conversations: List[Dict[str, Any]], run_folder: str):
    """Save results to local files."""
    # Save raw conversation data
    conversations_file = os.path.join(run_folder, "extracted_conversations.json")
    with open(conversations_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    print(f"✓ Saved {len(conversations)} conversations to {conversations_file}")
    
    # Create summary statistics
    summary = {
        'total_conversations': len(conversations),
        'tasks': list(set(conv.get('task') for conv in conversations if conv.get('task'))),
        'episodes_per_task': {},
        'total_messages': sum(len(conv.get('conversations', [])) for conv in conversations),
        'model': conversations[0].get('model') if conversations else None,
        'run_id': conversations[0].get('run_id') if conversations else None,
        'date_processed': datetime.now().isoformat()
    }
    
    # Count episodes per task
    for conv in conversations:
        task = conv.get('task')
        if task:
            summary['episodes_per_task'][task] = summary['episodes_per_task'].get(task, 0) + 1
    
    summary_file = os.path.join(run_folder, "analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved analysis summary to {summary_file}")

def main():
    """Main function to analyze terminal-bench run and optionally upload to Hugging Face."""
    run_folder = "/Users/etashguha/Documents/terminal-bench/runs/2025-07-24__09-58-13"
    
    if not os.path.exists(run_folder):
        print(f"Error: Run folder not found: {run_folder}")
        return
    
    print("Terminal Bench Run Analysis")
    print("=" * 50)
    
    # Process the run folder
    conversations = process_run_folder(run_folder)
    
    if not conversations:
        print("No conversation data extracted.")
        return
    
    # Save local results
    save_local_results(conversations, run_folder)
    
    # Display sample conversation
    if conversations:
        print("\nSample conversation:")
        sample = conversations[0]
        print(f"Task: {sample.get('task')}")
        print(f"Episode: {sample.get('episode')}")
        print(f"Model: {sample.get('model')}")
        print(f"Messages: {len(sample.get('conversations', []))}")
        if sample.get('conversations'):
            first_msg = sample['conversations'][0]
            print(f"First message role: {first_msg.get('role')}")
            print(f"First message preview: {first_msg.get('content', '')[:200]}...")
    
    # Try to upload to HuggingFace if token is available
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if hf_token:
        print("\nHuggingFace token found. Attempting upload...")
        success = upload_to_huggingface(conversations)
        if success:
            print("Upload completed successfully!")
        else:
            print("Upload failed - results saved locally only.")
    else:
        print("\nNo HuggingFace token found. Results saved locally only.")
        print("To upload, set HUGGINGFACE_TOKEN or HF_TOKEN environment variable.")

if __name__ == "__main__":
    main()