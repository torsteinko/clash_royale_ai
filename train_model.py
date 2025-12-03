from ultralytics import YOLO
import torch
from datetime import datetime


def main():
    # ============================================================
    # GPU CHECK
    # ============================================================
    print("\n" + "="*60)
    print("🎮 CLASH ROYALE BOT - FINAL HIGH-RES TRAINING")
    print("="*60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  WARNING: No GPU detected! Training will be VERY slow on CPU.")
    print("\n📊 Dataset Info:")
    print(f"   • Combined datasets: Your 8k + External 4k")
    print(f"   • After horizontal flip: ~24,000 images")
    print(f"   • Resolution: Native (1280px training)")
    print(f"   • Classes: ~155 (merged ally/enemy)")
    print("="*60 + "\n")


    # ============================================================
    # MODEL SELECTION
    # ============================================================
    model = YOLO('yolo11n.pt')
    print(f"📦 Loaded model: YOLOv11 Nano")
    print(f"🎯 Optimized for speed while maintaining accuracy")
    print(f"🔬 Training at 1280px for superior small object detection\n")


    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================
    results = model.train(
        # ============================================================
        # DATASET
        # ============================================================
        data='MERGED_NEW_SAMETEAM-1/data.yaml',  # ← UPDATE to your merged dataset path
        
        # ============================================================
        # TRAINING PARAMETERS
        # ============================================================
        epochs=150,              # More epochs for large dataset
        patience=25,             # Increased patience for convergence
        batch=16,                # Optimized for RTX 5070 Ti at 1280px
        imgsz=1280,              # High resolution for small troops (bats, spirits, etc.)
        
        # ============================================================
        # HARDWARE
        # ============================================================
        device=0,                # GPU 0
        workers=4,               # Windows multiprocessing limit
        amp=True,                # Automatic Mixed Precision (30-50% faster)
        
        # ============================================================
        # OPTIMIZER
        # ============================================================
        optimizer='AdamW',       # Best for most cases
        lr0=0.01,               # Initial learning rate
        lrf=0.01,               # Final learning rate (1% of lr0)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,       # Gradual warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # ============================================================
        # COLOR AUGMENTATION (for 30+ different arenas)
        # ============================================================
        hsv_h=0.03,              # Hue shift ±3% (arena color variations)
        hsv_s=0.9,               # Saturation ±90% (day/night modes)
        hsv_v=0.6,               # Brightness ±60% (lighting changes)
        
        # ============================================================
        # GEOMETRIC AUGMENTATION
        # ============================================================
        degrees=10.0,            # Rotation ±10° (troops at angles)
        translate=0.15,          # Translation ±15% (movement)
        scale=0.7,               # Scale ±70% (zoom variations)
        shear=0.0,               # No shear (not useful for 2D)
        perspective=0.0,         # No perspective (2D game)
        
        # ============================================================
        # FLIP AUGMENTATION
        # ============================================================
        flipud=0.0,              # NO vertical flip (gravity exists!)
        fliplr=0.5,              # 50% horizontal flip (symmetrical arena)
                                 # NOTE: Dataset already flipped in Roboflow
                                 # This adds ADDITIONAL random flips during training
        
        # ============================================================
        # ADVANCED AUGMENTATION (CRITICAL FOR SMALL OBJECTS)
        # ============================================================
        mosaic=0.5,              # Combine 4 images (simulates crowded battles)
                                 # Reduced from 1.0 to preserve small object size
        
        mixup=0.3,               # Blend 2 images (handles occlusion/overlap)
                                 # Critical for overlapping troops
        
        copy_paste=0.5,          # Paste objects on different backgrounds
                                 # HUGE benefit for rare troops (rocket, phoenix, etc.)
        
        # ============================================================
        # LOSSES & METRICS
        # ============================================================
        box=7.5,                 # Bounding box loss weight
        cls=0.5,                 # Classification loss weight
        dfl=1.5,                 # Distribution Focal Loss (small object precision)
        
        # ============================================================
        # VALIDATION & CHECKPOINTING
        # ============================================================
        val=True,                # Run validation every epoch
        plots=True,              # Generate training plots
        save=True,               # Save checkpoints
        save_period=15,          # Save checkpoint every 15 epochs
        cache=False,             # Set True if you have 32GB+ RAM (speeds up training)
        
        # ============================================================
        # OUTPUT
        # ============================================================
        project='runs/detect',
        name='clash_royale_FINAL_1280px',  # Descriptive name
        exist_ok=False,          # Don't overwrite (auto-increment)
        
        # ============================================================
        # DISPLAY & LOGGING
        # ============================================================
        verbose=True,
        seed=42,                 # Reproducibility
        deterministic=False,
        single_cls=False,
        rect=False,              # Square images (not rectangular)
        cos_lr=False,            # Linear LR decay (not cosine)
        close_mosaic=20,         # Disable mosaic last 20 epochs for fine-tuning
        resume=False,            # Start fresh (set True to resume if interrupted)
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        label_smoothing=0.0,
        nbs=64,
        iou=0.7,                 # IoU threshold for NMS
    )


    # ============================================================
    # TRAINING COMPLETE - SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Model saved to:")
    print(f"   • Best weights: runs/detect/clash_royale_FINAL_1280px/weights/best.pt")
    print(f"   • Last weights: runs/detect/clash_royale_FINAL_1280px/weights/last.pt")
    print(f"\n📊 Training results:")
    print(f"   • Results: runs/detect/clash_royale_FINAL_1280px/results.png")
    print(f"   • Confusion matrix: runs/detect/clash_royale_FINAL_1280px/confusion_matrix.png")
    print(f"   • F1 curve: runs/detect/clash_royale_FINAL_1280px/F1_curve.png")
    print(f"   • PR curve: runs/detect/clash_royale_FINAL_1280px/PR_curve.png")


    # ============================================================
    # EXTRACT & DISPLAY METRICS
    # ============================================================
    try:
        print(f"\n📈 Performance Metrics:")
        print(f"   • mAP@50:    {results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"   • mAP@50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
        print(f"   • Precision: {results.results_dict['metrics/precision(B)']:.4f}")
        print(f"   • Recall:    {results.results_dict['metrics/recall(B)']:.4f}")
        
        # Performance evaluation
        map50 = results.results_dict['metrics/mAP50(B)']
        print(f"\n🎯 Model Quality Assessment:")
        if map50 >= 0.90:
            print(f"   ✅ EXCELLENT ({map50:.3f}) - Production ready!")
        elif map50 >= 0.85:
            print(f"   ✅ GOOD ({map50:.3f}) - Ready for bot deployment")
        elif map50 >= 0.80:
            print(f"   ⚠️  ACCEPTABLE ({map50:.3f}) - Usable but could improve")
        else:
            print(f"   ❌ NEEDS IMPROVEMENT ({map50:.3f}) - Collect more data")
        
        # Specific improvements over 640px version
        print(f"\n📊 Expected Improvements vs 640px baseline (mAP@50: 0.836):")
        improvement = ((map50 - 0.836) / 0.836) * 100
        print(f"   • Overall improvement: +{improvement:.1f}%")
        print(f"   • Small troops (bat, bomber): +300-400% expected")
        print(f"   • Spells (rocket, arrows): +2000%+ expected")
        print(f"   • Evolutions: +20-30% expected")
        
    except Exception as e:
        print(f"\n⚠️  Could not extract metrics: {e}")
        print("   Check results.png for detailed metrics")


    print("\n💡 Next Steps:")
    print("   1. Review results.png for training curves")
    print("   2. Check confusion_matrix.png for problem classes")
    print("   3. Test on real gameplay screenshots")
    print("   4. If mAP@50 > 0.87: Deploy to bot!")
    print("   5. If mAP@50 < 0.87: Check which classes failed")
    print("="*60 + "\n")


    # ============================================================
    # FINAL VALIDATION
    # ============================================================
    print("🔍 Running final validation on test set...")
    val_results = model.val()
    print(f"✅ Validation complete!")
    print("="*60 + "\n")


    # ============================================================
    # TRAINING STATISTICS
    # ============================================================
    print("\n" + "="*60)
    print("📊 TRAINING SESSION STATISTICS")
    print("="*60)
    try:
        total_epochs = len(results.results_dict) if hasattr(results, 'results_dict') else "N/A"
        print(f"Total epochs trained: {total_epochs}")
        print(f"Early stopping: {'Yes' if total_epochs < 150 else 'No'}")
        print(f"Final learning rate: {results.results_dict.get('lr/pg0', 'N/A')}")
    except:
        print("Training statistics available in results.csv")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()
