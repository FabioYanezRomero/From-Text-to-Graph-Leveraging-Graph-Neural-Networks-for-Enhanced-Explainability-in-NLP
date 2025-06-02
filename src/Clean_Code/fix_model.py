#!/usr/bin/env python3
"""
Fix Model Script

This script creates a fixed model file for the RoBERTa CRF constituency parser.
It downloads the model from the official source and saves it in a format that can be loaded by our code.
"""

import os
import torch
import supar
from supar import CRFConstituencyParser

def fix_roberta_model():
    """Download and save the RoBERTa CRF constituency parser model in a usable format"""
    print("Fixing RoBERTa CRF constituency parser model...")
    
    # Create output directory
    os.makedirs('/app/models/fixed', exist_ok=True)
    output_path = '/app/models/fixed/roberta_crf_model.pt'
    
    try:
        # Create a new CRFConstituencyParser instance
        parser = CRFConstituencyParser.load('ptb.crf.con.roberta')
        
        # Save the model to the output path
        torch.save(parser, output_path)
        print(f"Model successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error fixing model: {e}")
        return False

if __name__ == "__main__":
    fix_roberta_model()
