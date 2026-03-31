A deep learning project for detecting and segmenting oil spills in satellite/aerial imagery.

Key Features:

Model: UNet with EfficientNet-B2 encoder and SCSE attention decoder
Multi-class segmentation: 3 classes (background, oil, other)
Dual loss function: Combination of weighted CrossEntropyLoss (emphasizing oil class) and TverskyLoss (optimizing for recall)
Data augmentation: Includes flips, rotations, color jittering, and gaussian blur
Training optimizations:
AdamW optimizer with cosine annealing learning rate scheduler
Early stopping (stops after 5 epochs without improvement)
Weighted class balancing (0.2, 1.2, 0.3) to emphasize oil detection
Inference:
Predicts segmentation masks on test images
Post-processes with connected components to remove small noise
Visualizes results with overlay on original images
Dataset: Multi-domain training data (combined images/masks) validated on separate validation set

Output: Trained model saved at final_multi_domain_model.pth with visualization of detection results

