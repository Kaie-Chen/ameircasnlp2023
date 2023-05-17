# LanguageDataset

from typing import Dict, List, Optional

class LanguageDataset:   
    def __init__(self, raw_data):
        self.data = raw_data
 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]