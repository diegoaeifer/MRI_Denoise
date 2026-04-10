import numpy as np

# Simulate what happens in dataset.py
image = np.random.randint(0, 65535, (128, 128)).astype(np.float32)

p_min, p_max = np.percentile(image, [0.05, 99.5])

# In-place clip
np.clip(image, p_min, p_max, out=image)

denom = float(p_max - p_min)

if denom > 1e-8:
    image -= p_min
    image /= denom

print("Success! Image dtype is:", image.dtype)
print("Min:", image.min(), "Max:", image.max())
