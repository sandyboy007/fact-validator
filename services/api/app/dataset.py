"""
Dataset management and splitting utilities for reproducible evaluation.

Ensures deterministic train/test splits with stratification.
"""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import random


@dataclass
class DatasetSplit:
    """Container for dataset split information."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    seed: int
    stratification_key: str
    

class DatasetManager:
    """Manage dataset loading, splitting, and versioning."""
    
    def __init__(self, dataset_path: str = "docs/evaluation_benchmark.json"):
        """
        Initialize dataset manager.
        
        Args:
            dataset_path: Path to evaluation benchmark JSON
        """
        self.dataset_path = Path(dataset_path)
        self.claims = self._load_dataset()
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from JSON file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        return data.get("claims", [])
    
    def get_dataset_info(self) -> Dict:
        """Get dataset statistics."""
        info = {
            "total_claims": len(self.claims),
            "by_category": {},
            "by_label": {},
            "by_difficulty": {}
        }
        
        for claim in self.claims:
            # Category
            category = claim.get("category", "unknown")
            info["by_category"][category] = info["by_category"].get(category, 0) + 1
            
            # Label
            label = claim.get("label", "unknown")
            info["by_label"][label] = info["by_label"].get(label, 0) + 1
            
            # Difficulty
            difficulty = claim.get("difficulty", "unknown")
            info["by_difficulty"][difficulty] = info["by_difficulty"].get(difficulty, 0) + 1
        
        return info
    
    def stratified_split(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stratify_by: str = "label",
        seed: int = 42
    ) -> DatasetSplit:
        """
        Create stratified train/val/test split.
        
        Maintains distribution of stratification key across splits.
        
        Args:
            train_ratio: Proportion for training (default 0.6)
            val_ratio: Proportion for validation (default 0.2)
            test_ratio: Proportion for testing (default 0.2)
            stratify_by: Column to stratify by ("label", "category", "difficulty")
            seed: Random seed for reproducibility
        
        Returns:
            DatasetSplit with train/val/test indices
        """
        random.seed(seed)
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"
        
        # Group by stratification key
        groups: Dict[str, List[int]] = {}
        for idx, claim in enumerate(self.claims):
            key = claim.get(stratify_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Split within each group to maintain proportions
        for group_key, indices in groups.items():
            # Shuffle within group
            random.shuffle(indices)
            
            n = len(indices)
            train_n = int(n * train_ratio)
            val_n = int(n * val_ratio)
            
            train_indices.extend(indices[:train_n])
            val_indices.extend(indices[train_n:train_n + val_n])
            test_indices.extend(indices[train_n + val_n:])
        
        # Final shuffle of each split
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)
        
        return DatasetSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            seed=seed,
            stratification_key=stratify_by
        )
    
    def get_split_data(
        self,
        split: DatasetSplit
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Get actual data for each split.
        
        Returns:
            (train_claims, val_claims, test_claims)
        """
        train = [self.claims[i] for i in split.train_indices]
        val = [self.claims[i] for i in split.val_indices]
        test = [self.claims[i] for i in split.test_indices]
        
        return train, val, test
    
    def verify_split_balance(
        self,
        split: DatasetSplit,
        stratify_by: str = "label",
        tolerance: float = 0.15
    ) -> Dict[str, bool]:
        """
        Verify that split maintains stratification balance.
        
        Args:
            split: DatasetSplit to verify
            stratify_by: Key used for stratification
            tolerance: Max allowed deviation from ideal ratio
        
        Returns:
            Dict with verification results
        """
        total = len(self.claims)
        ideal_train_ratio = len(split.train_indices) / total
        ideal_val_ratio = len(split.val_indices) / total
        ideal_test_ratio = len(split.test_indices) / total
        
        results = {"balanced": True, "details": {}}
        
        # Check each group
        groups: Dict[str, List[int]] = {}
        for idx, claim in enumerate(self.claims):
            key = claim.get(stratify_by, "unknown")
            if key not in groups:
                groups[key] = [0, 0, 0]  # [train, val, test]
            
            if idx in split.train_indices:
                groups[key][0] += 1
            elif idx in split.val_indices:
                groups[key][1] += 1
            elif idx in split.test_indices:
                groups[key][2] += 1
        
        for group_key, counts in groups.items():
            group_total = sum(counts)
            if group_total == 0:
                continue
            
            train_ratio = counts[0] / group_total
            val_ratio = counts[1] / group_total
            test_ratio = counts[2] / group_total
            
            train_balanced = abs(train_ratio - ideal_train_ratio) < tolerance
            val_balanced = abs(val_ratio - ideal_val_ratio) < tolerance
            test_balanced = abs(test_ratio - ideal_test_ratio) < tolerance
            
            balanced = train_balanced and val_balanced and test_balanced
            results["balanced"] = results["balanced"] and balanced
            
            results["details"][group_key] = {
                "counts": {"train": counts[0], "val": counts[1], "test": counts[2]},
                "ratios": {
                    "train": round(train_ratio, 3),
                    "val": round(val_ratio, 3),
                    "test": round(test_ratio, 3)
                },
                "balanced": balanced
            }
        
        return results
    
    def export_split(
        self,
        split: DatasetSplit,
        output_dir: str = "data/splits"
    ) -> Dict[str, str]:
        """
        Export split to JSON files for reproducibility.
        
        Args:
            split: DatasetSplit to export
            output_dir: Directory to save split files
        
        Returns:
            Dict with paths to exported files
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        train, val, test = self.get_split_data(split)
        
        # Export split metadata
        metadata = {
            "seed": split.seed,
            "stratification_key": split.stratification_key,
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
            "train_ratio": len(train) / len(self.claims),
            "val_ratio": len(val) / len(self.claims),
            "test_ratio": len(test) / len(self.claims)
        }
        
        paths = {}
        
        # Save train set
        train_path = out_path / "train.json"
        with open(train_path, 'w') as f:
            json.dump({"claims": train, "metadata": metadata}, f, indent=2)
        paths["train"] = str(train_path)
        
        # Save val set
        val_path = out_path / "val.json"
        with open(val_path, 'w') as f:
            json.dump({"claims": val, "metadata": metadata}, f, indent=2)
        paths["val"] = str(val_path)
        
        # Save test set
        test_path = out_path / "test.json"
        with open(test_path, 'w') as f:
            json.dump({"claims": test, "metadata": metadata}, f, indent=2)
        paths["test"] = str(test_path)
        
        # Save split info
        info_path = out_path / "split_info.json"
        with open(info_path, 'w') as f:
            split_info = {
                "metadata": metadata,
                "train_indices": split.train_indices,
                "val_indices": split.val_indices,
                "test_indices": split.test_indices,
                "balance_verification": {}  # Will be populated separately
            }
            json.dump(split_info, f, indent=2)
        paths["info"] = str(info_path)
        
        return paths


def create_standard_split(
    seed: int = 42,
    output_dir: str = "data/splits"
) -> DatasetSplit:
    """
    Create and export standard split for reproduction.
    
    Standard split:
    - 60% train, 20% val, 20% test
    - Stratified by label
    - Seed: 42
    
    Returns:
        DatasetSplit object
    """
    manager = DatasetManager()
    split = manager.stratified_split(seed=seed, stratify_by="label")
    
    # Verify balance
    balance = manager.verify_split_balance(split, stratify_by="label")
    print(f"Split balanced: {balance['balanced']}")
    
    # Export
    paths = manager.export_split(split, output_dir=output_dir)
    print(f"Split files exported to {output_dir}")
    
    return split
