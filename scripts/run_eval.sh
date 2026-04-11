# Same-dataset evaluation (image_dir_name comes from checkpoint config — no override needed)
python evaluate.py vit \
  --checkpoint checkpoints/vit/best_v6.pt \
  --data-root data/datasets/flickr30k \
  --batch-size 64 \
  --max-len 34 \
  --num-workers 0 \
  --device cuda \
  --seed 42

# Cross-dataset evaluation (e.g. flickr8k-trained model evaluated on flickr30k)
# python evaluate.py vit \
#   --checkpoint checkpoints/vit/best_v4.pt \
#   --data-root data/datasets/flickr30k \
#   --batch-size 64 \
#   --max-len 34 \
#   --num-workers 0 \
#   --device cuda \
#   --seed 42

# Cross-dataset evaluation (e.g. flickr30k-trained model evaluated on flickr8k)
# python evaluate.py vit \
#   --checkpoint checkpoints/vit/best_v6.pt \
#   --data-root data/datasets/flickr8k \
#   --batch-size 64 \
#   --max-len 34 \
#   --num-workers 0 \
#   --device cuda \
#   --seed 42
