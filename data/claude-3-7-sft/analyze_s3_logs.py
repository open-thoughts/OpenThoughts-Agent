#!/usr/bin/env python3

import boto3
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
import re
from datasets import Dataset
from huggingface_hub import HfApi
import os
import ray

def list_s3_folders(bucket: str, prefix: str = "") -> List[str]:
    """List folders in S3 bucket."""
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    folders = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
        if 'CommonPrefixes' in page:
            for obj in page['CommonPrefixes']:
                folders.add(obj['Prefix'].rstrip('/'))
    
    return sorted(list(folders))

@ray.remote
def check_folder_for_episodes(bucket, folder):
    """Check a single folder for episodes - designed for ray processing."""
    
    print(f"  Checking folder: {folder}")
    s3 = boto3.client('s3')
    
    try:
        # First get task folders within the run folder
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{folder}/",
            Delimiter='/'
        )
        
        if 'CommonPrefixes' in response:
            task_folders = [prefix['Prefix'] for prefix in response['CommonPrefixes']]
            print(f"    Found {len(task_folders)} task folders")
            
            all_episode_folders = []
            for task_folder in task_folders:
                # Check for agent-logs subfolder
                agent_logs_prefix = f"{task_folder}agent-logs/"
                logs_response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=agent_logs_prefix,
                    Delimiter='/'
                )
                
                if 'CommonPrefixes' in logs_response:
                    episode_folders = [
                        prefix['Prefix'] for prefix in logs_response['CommonPrefixes']
                        if re.search(r'episode-\d+', prefix['Prefix'])
                    ]
                    print(f"    Found {len(episode_folders)} episode folders in {task_folder}")
                    
                    # Keep only the last episode from this task
                    if episode_folders:
                        def extract_episode_number(folder_path):
                            match = re.search(r'episode-(\d+)', folder_path)
                            return int(match.group(1)) if match else 0
                        
                        # sorted_episodes = sorted(episode_folders, key=extract_episode_number)
                        # last_episode = sorted_episodes[-1]
                        all_episode_folders.extend(episode_folders)
            
            if all_episode_folders:
                print(f"  ✓ Folder {folder} has {len(all_episode_folders)} episodes")
                return (folder, all_episode_folders)
            else:
                print(f"  ✗ Folder {folder} has no episodes")
                return None
                
    except Exception as e:
        print(f"Error checking folder {folder}: {e}")
        return None

def get_recent_folders_with_episodes(bucket: str) -> List[Tuple[str, List[str]]]:
    """Get most recent folders that contain episode subfolders."""
    folders = list_s3_folders(bucket)
    print(f"Found {len(folders)} total folders")
    
    # Focus on March 2025 data where episodes exist
    folders_with_data = [f for f in folders if f.startswith('2025')]
    if not folders_with_data:
        folders_with_data = sorted(folders, reverse=True)[:20]
    folders_with_data = sorted(folders_with_data, reverse=True)
    
    print(f"Checking {len(folders_with_data)} folders from {folders_with_data[0] if folders_with_data else 'N/A'} to {folders_with_data[-1] if folders_with_data else 'N/A'}")
    
    # Use ray to check folders in parallel
    print(f"Processing {len(folders_with_data)} folders in parallel...")
    
    results = ray.get([check_folder_for_episodes.remote(bucket, folder) for folder in folders_with_data])
    # Filter out None results
    folders_with_episodes = [result for result in results if result is not None]
    
    # Sort by folder name (assuming timestamp-based naming)
    folders_with_episodes.sort(key=lambda x: x[0], reverse=True)
    return folders_with_episodes

def get_all_episode_files(bucket: str, folder_path: str, episode_folders: List[str]) -> List[str]:
    """Get files from all episode folders."""
    if not episode_folders:
        return []
    
    s3 = boto3.client('s3')
    all_files = []
    
    # Get files from all episodes (now filtered to just the last episode at line 92)
    for ep_folder in episode_folders:
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=ep_folder
        )
        
        if 'Contents' in response:
            episode_files = [obj['Key'] for obj in response['Contents']]
            all_files.extend(episode_files)
    
    return all_files

def filter_terminus_claude_files(files: List[str]) -> List[str]:
    """Filter files for terminus agent and claude model."""
    # Only need debug.json files which contain both input and output data
    filtered_files = []
    
    for file in files:
        if file.endswith('debug.json'):
            filtered_files.append(file)
    
    return filtered_files

def extract_conversation_data(bucket: str, debug_file: str) -> Dict[str, Any]:
    """Extract conversation data from debug.json and response.json files."""
    s3 = boto3.client('s3')
    
    conversation_data = {
        'conversations': [],
        'agent': None,
        'model': None,
        'date': None,
        'task': None,
        'episode': None
    }
    
    try:
        # Extract date from file path (e.g., "2025-03-31__17-15-21")
        path_parts = debug_file.split('/')
        if len(path_parts) > 0:
            date_part = path_parts[0]
            if '__' in date_part:
                date_str = date_part.replace('__', ' ')
                try:
                    conversation_data['date'] = datetime.strptime(date_str, '%Y-%m-%d %H-%M-%S').isoformat()
                except:
                    conversation_data['date'] = date_str
            
            # Extract task name
            if len(path_parts) > 1:
                conversation_data['task'] = path_parts[1]
            
            # Extract episode number
            if len(path_parts) > 3:
                episode_part = path_parts[3]
                if 'episode-' in episode_part:
                    conversation_data['episode'] = episode_part
        
        # Read debug.json for both model info and full conversation
        try:
            debug_response = s3.get_object(Bucket=bucket, Key=debug_file)
            debug_content = debug_response['Body'].read().decode('utf-8')
            debug_data = json.loads(debug_content)
            
            # Extract model name
            if 'model' in debug_data:
                conversation_data['model'] = debug_data['model']
            
            # Extract full conversation from 'input' field which contains the messages
            if 'input' in debug_data and isinstance(debug_data['input'], list):
                messages = debug_data['input']
                for msg in messages:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        conversation_data['conversations'].append({
                            'role': msg['role'],
                            'content': msg['content']
                        })
            
            # Extract original response and add as assistant message
            if 'original_response' in debug_data:
                original_response = debug_data['original_response']
                if original_response:  # Only add if response is not empty
                    try:
                        data = json.loads(original_response)
                    except Exception as e:
                       return None
                    if 'content' in data:
                        if "values" not in data['content'][0]['input']:
                            return None
                        conversation_data['conversations'].append({
                            'role': 'assistant',
                            'content': str(data['content'][0]['input']['values'])
                        })
                    elif 'choices' in data:
                        conversation_data['conversations'].append({
                            'role': 'assistant',
                            'content': str(data['choices'][0]['message']['content'])
                        })
                    elif 'candidates' in data:
                        conversation_data['conversations'].append({
                            'role': 'assistant',
                            'content': str(data['candidates'][0]['content']['parts'][0]['text'])
                        })
                    else:
                        print(f"Error extracting original response: {data}")
                        breakpoint()
            
        except Exception as e:
            breakpoint()
            print(f"Error reading debug file {debug_file}: {e}")
        
        # Set agent type based on model and context
        model_str = str(conversation_data['model']).lower()
        if 'claude' in model_str:
            conversation_data['agent'] = 'terminus'  # Assuming terminus agent for Claude models
        elif 'qwen' in model_str or 'gpt' in model_str or 'llama' in model_str:
            conversation_data['agent'] = 'terminus'  # Terminal-bench likely uses terminus agent
        else:
            conversation_data['agent'] = 'terminus'  # Default to terminus since this is terminal-bench data
        
        return conversation_data
        
    except Exception as e:
        print(f"Error extracting conversation data: {e}")
        return conversation_data

@ray.remote
def process_debug_files_batch(bucket_name, debug_files_batch):
    """Process a batch of debug files - designed for ray processing."""
    batch_conversations = []
    
    for debug_file in debug_files_batch:
        try:
            conversation_data = extract_conversation_data(bucket_name, debug_file)
            if conversation_data is None:
                continue
            if conversation_data['conversations']:  # Only add if we have conversation data
                batch_conversations.append(conversation_data)
        except Exception as e:
            print(f"Error processing {debug_file}: {e}")
            continue
    
    return batch_conversations

@ray.remote
def process_folder(bucket_name, folder, episode_folders):
    """Process a single folder - designed for ray processing."""
    
    print(f"\nProcessing folder: {folder}")
    
    # Get files from all episodes (now filtered to just the last episode at line 92)
    episode_files = get_all_episode_files(bucket_name, folder, episode_folders)
    
    # Filter for terminus + claude files
    terminus_claude_files = filter_terminus_claude_files(episode_files)
    
    if not terminus_claude_files:
        print(f"No terminus + claude files found in {folder}")
        return []
    
    print(f"Found {len(terminus_claude_files)} debug.json files")
    print(f"Found {len(episode_files)} total episode files")
    
    # Process debug.json files in batches using multiprocessing
    batch_size = max(1, len(terminus_claude_files) // (8 * 4))
    if batch_size > 50:  # Limit batch size for memory management
        batch_size = 50
        
    batches = [terminus_claude_files[i:i + batch_size] for i in range(0, len(terminus_claude_files), batch_size)]
    
    print(f"Processing {len(terminus_claude_files)} debug files in {len(batches)} batches...")
    
    # Use ray to process debug files in parallel
    results = ray.get([process_debug_files_batch.remote(bucket_name, batch) for batch in batches])
    
    # Flatten results
    folder_conversations = []
    for result in results:
        folder_conversations.extend(result)
    
    return folder_conversations

def upload_to_huggingface(dataset_rows: List[Dict], repo_id: str = "mlfoundations-dev/test-tbench-traces"):
    """Upload dataset to Hugging Face repository."""
    if not dataset_rows:
        print("No data to upload.")
        return
    
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
                    print(f"    First message: {str(value[0])[:100]}...")
            else:
                print(f"  {key}: {str(value)[:100]}")
    
    # Upload to Hugging Face
    try:
        dataset.push_to_hub(repo_id, token=hf_token)
        print(f"✓ Successfully uploaded dataset to {repo_id}")
    except Exception as e:
        print(f"✗ Error uploading to Hugging Face: {e}")
        return False
    
    return True

def main():
    """Main function to analyze S3 logs and upload to Hugging Face."""
    ray.init()
    
    try:
        bucket_name = "t-bench-mam"
    
        print("Getting recent folders with episodes...")
        folders_with_episodes = get_recent_folders_with_episodes(bucket_name)
        if not folders_with_episodes:
            print("No folders with episodes found.")
            return
        
        print(f"Found {len(folders_with_episodes)} folders with episodes.")
        
        all_conversation_data = []
    
        # Process ALL folders with episodes for complete dataset using ray
        print(f"Processing {len(folders_with_episodes)} folders in parallel...")
        
        results = ray.get([process_folder.remote(bucket_name, folder, episode_folders) for folder, episode_folders in folders_with_episodes])
        
        # Flatten results
        for result in results:
            all_conversation_data.extend(result)
        
        print(f"\nTotal conversations extracted: {len(all_conversation_data)}")
        
        if all_conversation_data:
            # Display first example
            print("\nFirst conversation example:")
            sample = all_conversation_data[0]
            print(f"Agent: {sample['agent']}")
            print(f"Model: {sample['model']}")
            print(f"Date: {sample['date']}")
            print(f"Task: {sample['task']}")
            print(f"Episode: {sample['episode']}")
            print(f"Messages: {len(sample['conversations'])}")
            
            # Upload to Hugging Face
            success = upload_to_huggingface(all_conversation_data)
            if success:
                print("Upload completed successfully!")
            else:
                print("Upload failed.")
        else:
            print("No conversation data extracted.")
        
        return all_conversation_data
    finally:
        ray.shutdown()

if __name__ == "__main__":
    messages_outputs = main()