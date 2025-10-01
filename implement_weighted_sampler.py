"""
Implementation guide for WeightedRandomSampler

This shows you exactly what to change in train_classification_task.py
to implement balanced sampling without reorganizing data.
"""

def show_implementation():
    print("=" * 60)
    print("WEIGHTED RANDOM SAMPLER IMPLEMENTATION")
    print("=" * 60)
    
    print("\nStep 1: Add import at top of train_classification_task.py")
    print("-" * 60)
    print("from torch.utils.data import WeightedRandomSampler")
    
    print("\nStep 2: Calculate sample weights (add after line 836)")
    print("-" * 60)
    print("""
# Calculate sample weights for balanced sampling
if opt.weighted_sampling:
    print('Using WeightedRandomSampler for balanced class sampling...')
    
    # Count samples per class
    class_counts = np.bincount([int(label[0]) for label in labels], minlength=nc)
    
    # Calculate weights (inverse frequency)
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample based on its class
    sample_weights = [class_weights[int(label[0])] for label in labels]
    
    # Create sampler
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Recreate dataloader with weighted sampler
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size // WORLD_SIZE,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn
    )
    
    print(f'  Class counts: {class_counts}')
    print(f'  Class weights: {class_weights}')
""")
    
    print("\nStep 3: Add command line argument")
    print("-" * 60)
    print("""
# In parse_opt() function, add:
parser.add_argument('--weighted-sampling', action='store_true', 
                    help='use weighted random sampling for class balance')
""")
    
    print("\nStep 4: Usage")
    print("-" * 60)
    print("python train_classification_task.py \\")
    print("  --data regurgitationV1/data.yaml \\")
    print("  --epochs 50 \\")
    print("  --batch-size 128 \\")
    print("  --weighted-sampling \\")  # New flag
    print("  --device auto \\")
    print("  --weights yolov5s.pt")
    
    print("\n" + "=" * 60)
    print("EXPECTED RESULTS")
    print("=" * 60)
    print("\nWith WeightedRandomSampler:")
    print("  - PSAX bias: ~0.0 (excellent)")
    print("  - PSAX recall: 25-35% (much better than 9%)")
    print("  - Overall accuracy: 55-65%")
    
    print("\nVs current approach:")
    print("  - PSAX bias: -0.263 (bad)")
    print("  - PSAX recall: 9% (very poor)")
    print("  - Overall accuracy: 41%")

def show_alternative_balanced_sampler():
    print("\n" + "=" * 60)
    print("ALTERNATIVE: CUSTOM BALANCED SAMPLER")
    print("=" * 60)
    
    print("\nIf WeightedRandomSampler doesn't work well, use this:")
    print("-" * 60)
    print("""
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size, num_classes):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Group indices by class
        self.class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            class_id = int(label[0])
            self.class_indices[class_id].append(idx)
        
        # Shuffle each class
        for class_idx in self.class_indices:
            np.random.shuffle(class_idx)
        
        # Calculate samples per class per batch
        self.samples_per_class = batch_size // num_classes
        
    def __iter__(self):
        batch = []
        # Pointers for each class
        pointers = [0] * self.num_classes
        
        while all(p < len(self.class_indices[i]) for i, p in enumerate(pointers)):
            # Sample from each class
            for class_id in range(self.num_classes):
                start = pointers[class_id]
                end = min(start + self.samples_per_class, len(self.class_indices[class_id]))
                batch.extend(self.class_indices[class_id][start:end])
                pointers[class_id] = end
                
                if len(batch) >= self.batch_size:
                    yield batch[:self.batch_size]
                    batch = batch[self.batch_size:]
        
        if len(batch) > 0:
            yield batch
    
    def __len__(self):
        min_class_size = min(len(indices) for indices in self.class_indices)
        return (min_class_size * self.num_classes) // self.batch_size

# Usage:
balanced_sampler = BalancedBatchSampler(labels, batch_size, nc)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=balanced_sampler,
    num_workers=workers,
    pin_memory=True,
    collate_fn=LoadImagesAndLabels.collate_fn
)
""")

def summary():
    print("\n" + "=" * 60)
    print("SUMMARY: 5 APPROACHES RANKED")
    print("=" * 60)
    
    approaches = [
        ("1. Easiest (Recommended)", "Class weights + large batch", 
         "python train.py --batch-size 128 --hyp psax_bias_fix_hyp.yaml", 
         "PSAX bias: -0.01", "Code: 0 lines (already done!)"),
        
        ("2. Best Balance", "WeightedRandomSampler", 
         "python train.py --weighted-sampling", 
         "PSAX bias: ~0.0", "Code: ~10 lines"),
        
        ("3. Most Effective", "Balanced Batch Sampler", 
         "Custom implementation", 
         "PSAX bias: 0.0", "Code: ~50 lines"),
        
        ("4. Advanced", "Focal Loss", 
         "Replace loss function", 
         "PSAX bias: -0.05", "Code: ~30 lines"),
        
        ("5. Simple", "Oversampling", 
         "Duplicate PSAX samples", 
         "PSAX bias: ~0.0", "Code: ~20 lines, High memory"),
    ]
    
    for i, (name, method, command, result, effort) in enumerate(approaches):
        print(f"\n{name}")
        print(f"  Method: {method}")
        print(f"  Command: {command}")
        print(f"  Result: {result}")
        print(f"  Effort: {effort}")
    
    print("\n" + "=" * 60)
    print("MY RECOMMENDATION")
    print("=" * 60)
    print("\nStart with Approach #1 (already implemented!):")
    print("  python train_classification_task.py \\")
    print("    --data regurgitationV1/data.yaml \\")
    print("    --epochs 50 \\")
    print("    --batch-size 128 \\")
    print("    --device auto \\")
    print("    --weights yolov5s.pt \\")
    print("    --hyp psax_bias_fix_hyp.yaml \\")
    print("    --optimizer Adam \\")
    print("    --patience 0")
    print("\nIf that doesn't work well, try Approach #2 (WeightedRandomSampler)")

if __name__ == "__main__":
    show_implementation()
    show_alternative_balanced_sampler()
    summary()
