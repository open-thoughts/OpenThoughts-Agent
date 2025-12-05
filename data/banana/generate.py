import datasets

def main() -> None:
    dataset = datasets.load_dataset("DCAgent/taskmaster2-gpt5mini")
    
    def modify_conversations(example):
        conversations = example['conversations']
        for conversation in conversations:
            if conversation.get('role') == 'assistant':
                # Replace assistant response with repeated "banana"
                original_length = len(conversation.get('content', ''))
                # Repeat "banana " to approximate original length
                num_bananas = max(1, original_length // 7)  # 7 is length of "banana "
                conversation['content'] = "banana " * num_bananas
        return example
    
    # Apply the modification to the dataset
    modified_dataset = dataset.map(modify_conversations)
    
    # Save the modified dataset
    modified_dataset.push_to_hub("DCAgent/taskmaster2-banana")

if __name__ == "__main__":
    main()