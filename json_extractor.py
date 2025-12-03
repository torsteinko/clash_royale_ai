import json
import requests
from pathlib import Path
import time
from PIL import Image
from io import BytesIO
import re

# Read the JSON file
with open('response.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create output directory structure
output_dir = Path('card_images')
output_dir.mkdir(exist_ok=True)

# Subdirectories
(output_dir / 'base').mkdir(exist_ok=True)
(output_dir / 'evolution').mkdir(exist_ok=True)

# Function to sanitize filename
def sanitize_filename(name):
    """Convert card name to valid filename"""
    # Remove special characters and replace spaces with underscores
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name

# Function to download image
def download_image(url, filepath):
    """Download image and return success status"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Check if it's actually an image
            img = Image.open(BytesIO(response.content))
            img.save(filepath)
            return True, img.size
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

# Statistics
total_cards = 0
successful_base = 0
successful_evo = 0
failed_downloads = []
cards_with_evolution = []
cards_without_evolution = []

print("=" * 80)
print("🎴 CLASH ROYALE CARD IMAGE DOWNLOADER")
print("=" * 80)
print(f"\n📁 Output directory: {output_dir.absolute()}\n")
print("Starting download...\n")

# Process all cards
for item in data['items']:
    total_cards += 1
    card_name = item['name']
    card_id = item['id']
    elixir_cost = item.get('elixirCost', 'N/A')
    rarity = item.get('rarity', 'unknown')
    
    # Sanitize name for filename
    filename_base = sanitize_filename(card_name)
    
    # Download base image
    base_url = item['iconUrls'].get('medium')
    if base_url:
        base_filepath = output_dir / 'base' / f"{filename_base}.png"
        success, result = download_image(base_url, base_filepath)
        
        if success:
            successful_base += 1
            size = result
            print(f"✅ {card_name:<25} → base/{filename_base}.png ({size[0]}x{size[1]})")
        else:
            failed_downloads.append({
                'card': card_name,
                'type': 'base',
                'url': base_url,
                'error': result
            })
            print(f"❌ {card_name:<25} → FAILED (base): {result}")
    
    # Download evolution image (if exists)
    evo_url = item['iconUrls'].get('evolutionMedium')
    if evo_url:
        cards_with_evolution.append({
            'name': card_name,
            'elixir': elixir_cost
        })
        
        evo_filepath = output_dir / 'evolution' / f"{filename_base}_evo.png"
        success, result = download_image(evo_url, evo_filepath)
        
        if success:
            successful_evo += 1
            size = result
            print(f"   ✨ Evolution → evolution/{filename_base}_evo.png ({size[0]}x{size[1]})")
        else:
            failed_downloads.append({
                'card': card_name,
                'type': 'evolution',
                'url': evo_url,
                'error': result
            })
            print(f"   ❌ Evolution → FAILED: {result}")
    else:
        cards_without_evolution.append({
            'name': card_name,
            'elixir': elixir_cost
        })
    
    # Small delay to avoid rate limiting
    time.sleep(0.1)

print("\n" + "=" * 80)
print("📊 DOWNLOAD SUMMARY")
print("=" * 80)
print(f"Total cards processed: {total_cards}")
print(f"✅ Base images downloaded: {successful_base}/{total_cards}")
print(f"✨ Evolution images downloaded: {successful_evo}/{len(cards_with_evolution)}")
print(f"❌ Failed downloads: {len(failed_downloads)}")

print("\n" + "=" * 80)
print("📈 STATISTICS")
print("=" * 80)
print(f"Cards with evolution: {len(cards_with_evolution)}")
print(f"Cards without evolution: {len(cards_without_evolution)}")

# Save statistics
print(f"\nCards with evolution/hero variants:")
print(f"{'Name':<25} {'Elixir':<10} {'Rarity':<15} {'Max Evo Level'}")
print("-" * 70)
for card in sorted(cards_with_evolution, key=lambda x: x['name']):
    print(f"{card['name']:<25} {str(card['elixir']):<10} {card['maxEvoLevel']}")

if failed_downloads:
    print("\n" + "=" * 80)
    print("❌ FAILED DOWNLOADS")
    print("=" * 80)
    for fail in failed_downloads:
        print(f"\n🔴 {fail['card']} ({fail['type']})")
        print(f"   URL: {fail['url']}")
        print(f"   Error: {fail['error']}")

print("\n" + "=" * 80)
print("✅ DOWNLOAD COMPLETE!")
print("=" * 80)