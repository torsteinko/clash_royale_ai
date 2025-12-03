# normalize_card_templates_v2.py
"""
IMPROVED Card Image Normalizer
- Handles different aspect ratios (hero vs base cards)
- Uses absolute pixel cropping instead of percentage
- Better border detection
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


class SmartCardNormalizer:
    """Smart normalizer that adapts to different card sizes"""
    
    def __init__(self, 
                 input_dir='card_images',
                 output_dir='card_templates_normalized',
                 target_size=(120, 90)):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # Create output directories
        for subfolder in ['base', 'evolution', 'hero']:
            (self.output_dir / subfolder).mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'processed': 0,
            'failed': []
        }
    
    def detect_card_content(self, img: np.ndarray) -> tuple:
        """
        Smart border detection that works for different card sizes
        
        Returns:
            (x, y, w, h): Bounding box of card content
        """
        height, width = img.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Method 1: Detect non-border colors
        # Borders are either very dark (black) or very bright/saturated (golden)
        
        # Create mask for card content (not border)
        # Exclude very dark pixels (black border)
        dark_mask = hsv[:, :, 2] > 30  # V channel > 30
        
        # Exclude highly saturated bright pixels (golden border glow)
        golden_mask = ~((hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 200))
        
        # Combine masks
        content_mask = dark_mask & golden_mask
        
        # Find bounding box of content
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)
        
        if rows.any() and cols.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Add small padding (5px) to avoid cutting into content
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Method 2: Fallback - use fixed pixel cropping
        # Hero cards: 285x342 → crop ~30px from each side
        # Base cards: 285x420 → crop ~35px from each side
        
        if height < 400:  # Likely hero card (342px)
            crop_pixels = 25
        else:  # Likely base/evo card (420px)
            crop_pixels = 30
        
        x_min = crop_pixels
        y_min = crop_pixels
        w = width - (crop_pixels * 2)
        h = height - (crop_pixels * 2)
        
        return (x_min, y_min, w, h)
    
    def normalize_card(self, img_path: Path, output_path: Path) -> tuple:
        """
        Normalize a single card image
        
        Returns:
            (success: bool, message: str)
        """
        try:
            # Load image
            img = cv2.imread(str(img_path))
            
            if img is None:
                return False, "Failed to load"
            
            orig_height, orig_width = img.shape[:2]
            
            # Detect content area
            x, y, w, h = self.detect_card_content(img)
            
            # Validate bounding box
            if w <= 0 or h <= 0:
                return False, "Invalid crop region"
            
            # Crop to content
            cropped = img[y:y+h, x:x+w]
            
            # Resize to target size
            resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Save
            cv2.imwrite(str(output_path), resized)
            
            return True, f"{orig_width}x{orig_height} → {w}x{h} → {self.target_size[0]}x{self.target_size[1]}"
            
        except Exception as e:
            return False, str(e)[:50]
    
    def process_folder(self, subfolder: str):
        """Process all images in a subfolder"""
        
        input_folder = self.input_dir / subfolder
        output_folder = self.output_dir / subfolder
        
        if not input_folder.exists():
            print(f"⚠️  Skipping {subfolder}/ (not found)")
            return
        
        images = list(input_folder.glob('*.png')) + list(input_folder.glob('*.jpg'))
        
        if not images:
            print(f"⚠️  No images in {subfolder}/")
            return
        
        print(f"\n{'='*70}")
        print(f"📁 Processing {subfolder}/ ({len(images)} images)")
        print(f"{'='*70}")
        
        for img_path in images:
            output_path = output_folder / f"{img_path.stem}.png"
            
            success, message = self.normalize_card(img_path, output_path)
            
            if success:
                self.stats['processed'] += 1
                print(f"  ✅ {img_path.name:<35} {message}")
            else:
                self.stats['failed'].append(img_path.name)
                print(f"  ❌ {img_path.name:<35} FAILED: {message}")
    
    def run(self):
        """Process all card images"""
        
        print("\n" + "="*80)
        print("🎴 SMART CARD IMAGE NORMALIZER v2.0")
        print("="*80)
        print(f"📥 Input:  {self.input_dir.absolute()}")
        print(f"📤 Output: {self.output_dir.absolute()}")
        print(f"📐 Target: {self.target_size[0]}x{self.target_size[1]} pixels")
        print("="*80)
        
        # Process each subfolder
        for subfolder in ['base', 'evolution', 'hero']:
            self.process_folder(subfolder)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary"""
        
        print("\n" + "="*80)
        print("📊 NORMALIZATION SUMMARY")
        print("="*80)
        print(f"\n✅ Successfully processed: {self.stats['processed']} images")
        
        if self.stats['failed']:
            print(f"❌ Failed: {len(self.stats['failed'])} images")
            for name in self.stats['failed']:
                print(f"   • {name}")
        
        print(f"\n{'='*80}")
        print("✅ NORMALIZATION COMPLETE!")
        print("="*80)
        print(f"\n📁 Templates saved to: {self.output_dir.absolute()}")
        print("="*80 + "\n")


# ========== COMPARISON TOOL ==========

def create_comparison(original_path: str, normalized_path: str, output_path: str):
    """Create comparison image"""
    
    orig = cv2.imread(original_path)
    norm = cv2.imread(normalized_path)
    
    if orig is None or norm is None:
        print(f"❌ Could not load images")
        return
    
    # Resize normalized to match original height
    h = orig.shape[0]
    w = int(norm.shape[1] * (h / norm.shape[0]))
    norm_resized = cv2.resize(norm, (w, h))
    
    # Add labels
    cv2.putText(orig, "ORIGINAL", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(norm_resized, "NORMALIZED", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Concatenate
    comparison = np.hstack([orig, norm_resized])
    
    cv2.imwrite(output_path, comparison)
    print(f"✅ Comparison saved: {output_path}")


# ========== MAIN ==========

if __name__ == '__main__':
    # Normalize all cards
    normalizer = SmartCardNormalizer(
        input_dir='card_images',
        output_dir='card_templates_normalized',
        target_size=(120, 90)
    )
    normalizer.run()
    
    # Create comparisons
    print("\n📸 Creating comparison images...")
    
    # Baby Dragon
    if Path('card_images/base/baby_dragon.png').exists():
        create_comparison(
            'card_images/base/baby_dragon.png',
            'card_templates_normalized/base/baby_dragon.png',
            'comparison_baby_dragon_v2.png'
        )
    
    # Mini PEKKA Hero
    if Path('card_images/hero/mini_pekka_hero.png').exists():
        create_comparison(
            'card_images/hero/mini_pekka_hero.png',
            'card_templates_normalized/hero/mini_pekka_hero.png',
            'comparison_mini_pekka_hero_v2.png'
        )
    
    print("\n✅ Done! Check the comparison images.")
