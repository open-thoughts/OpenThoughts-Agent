#!/bin/bash

# Script that reads model names from stdin and processes them
# Usage: cat model_list.txt | ./process_models.sh
# Or: echo -e "model1\nmodel2" | ./process_models.sh
# Or with heredoc: 
# ./goscale.sh << 'EOF'
# b2_math_random
# b2_math_embedding
# b2_math_fasttext
# b2_math_askllm
# b2_math_length
# b2_math_length_gpt4omini
# EOF

# Read each line from stdin
while IFS= read -r model; do
  # Skip empty lines
  [[ -z "$model" ]] && continue
  
  # Process each model with the different commands
  gomicro "${model}_0.3k"
  gosmall "${model}_1k"
  gosmall "${model}_3k"
  gotrain "${model}_10k"
done