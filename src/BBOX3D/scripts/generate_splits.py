import os
import random
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Generate splits for 3D bounding box dataset')
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to the dataset root directory'
        )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to save the generated splits')
    parser.add_argument(
        '--split_ratio',
        nargs='+',
        type = float,
        default=[0.8, 0.1, 0.1],
        help='Ratio of training, validation, and test sets in the total dataset'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    dirs_list = sorted(os.listdir(args.data_root))
    total_samples = len(dirs_list)
    train_samples = int(total_samples * args.split_ratio[0])
    val_samples = int(total_samples * args.split_ratio[1])
    test_samples = total_samples - train_samples - val_samples

    # Print statistics
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Test samples: {test_samples}")
    
    # Shuffle the directories
    random.shuffle(dirs_list)
    
    # Store the split dirs in yaml file
    split_dirs = {
        'train': dirs_list[:train_samples],
        'val': dirs_list[train_samples:train_samples+val_samples],
        'test': dirs_list[train_samples+val_samples:]
    }
    
    with open(os.path.join(args.output_dir, 'splits.yaml'), 'w') as f:
        yaml.dump(split_dirs, f)
    
    print(f"Splits saved to {os.path.join(args.output_dir, 'splits.yaml')}")
    
if __name__ == '__main__':
    main()
    
    
    
    
    


    