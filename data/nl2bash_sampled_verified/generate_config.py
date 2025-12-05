#!/usr/bin/env python3
"""Generate a sampling configuration for distribution-aware nl2bash data generation.

Run from project root as:
    python -m data.nl2bash.generate_config
"""

from pathlib import Path

from data.nl2bash import sampler

# Define target skill frequencies
# Note: NL2SH-ALFA dataset is bash-focused, so only CLI utilities are included
TARGETS = {
    # Core file operations
    "ls": 5,
    "cat": 5,
    "rm": 5,
    "cp": 5,
    "mv": 5,
    "mkdir": 4,
    "touch": 4,
    "chmod": 4,
    "ln": 3,
    "rmdir": 3,
    
    # Text processing
    "grep": 5,
    "awk": 4,
    "sed": 4,
    "cut": 3,
    "sort": 3,
    "uniq": 3,
    "head": 4,
    "tail": 4,
    "tr": 2,
    "paste": 2,
    
    # File searching
    "find": 4,
    "xargs": 3,
    
    # Pattern matching
    "POSIX regular expressions": 4,
    "egrep": 3,
    
    # System utilities
    "ps": 3,
    "date": 4,
    "env": 3,
    "which": 3,
    "lsof": 2,
    "chown": 2,
    
    # Archiving
    "tar": 3,
    
    # Network utilities
    "curl": 4,
    
    # Package management
    "apt": 3,
    
    # Misc utilities
    "base64": 3,
    "sha256sum": 3,
    "tee": 3,
    "openssl": 2,
    
    # JSON processing
    "jq": 4,
    
    # Output utilities
    "printf": 4,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate sampling config for nl2bash")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Target number of samples (default: minimum to satisfy targets)"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--max-overshoot",
        type=float,
        default=2.0,
        help="Maximum overshoot multiplier (e.g., 2.0 = allow up to 2x target)"
    )
    args = parser.parse_args()
    
    # Generate sampling config
    print("Generating sampling configuration...")
    config = sampler.generate_sampling_config(
        targets=TARGETS,
        split=args.split,
        seed=args.seed,
        num_samples=args.num_samples,
        max_overshoot=args.max_overshoot,
    )
    
    # Save to file
    output_path = Path("data/nl2bash_sampled_verified/sampling_config.json")
    sampler.save_config(config, output_path)
    
    # Print summary
    print(f"\n✓ Config saved to: {output_path}")
    print(f"✓ Total samples: {config.total_samples}")
    print(f"✓ Skills covered: {len(config.skill_coverage)}")
    
    # Print skill distribution
    print(f"\n=== SKILL DISTRIBUTION ===")
    sorted_skills = sorted(config.skill_coverage.items(), key=lambda x: -x[1])
    for skill, count in sorted_skills:
        target = TARGETS.get(skill, 0)
        if target > 0:
            status = "✓" if count >= target else "⚠"
            print(f"{status} {skill}: {count}/{target}")
        else:
            print(f"  {skill}: {count}")
    
    if config.unmet_targets:
        print(f"\n⚠ UNMET TARGETS ({len(config.unmet_targets)}):")
        for skill, remaining in sorted(config.unmet_targets.items(), key=lambda x: -x[1])[:10]:
            print(f"  - {skill}: {remaining} still needed")
    
    print(f"\n=== NEXT STEP ===")
    print(f"./data/nl2bash_sampled_verified/local_nl2bash.sh {config.total_samples} 1 1 {output_path}")


if __name__ == "__main__":
    main()

