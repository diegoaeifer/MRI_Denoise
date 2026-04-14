## 2024-05-24 - NumPy Array Normalization Bottlenecks in Data Loading
**Learning:** In PyTorch Datasets handling large images (like MRI DICOMs), seemingly harmless consecutive NumPy operations (`np.percentile`, `np.clip`, `.min()`, `.max()`, and array arithmetic) create massive overhead by repeatedly scanning arrays and allocating intermediate memory. Computing multiple percentiles in a single call, using `out=` arguments for in-place modifications, and reusing known bounds after clipping drastically reduces data loading times.
**Action:** Always batch `np.percentile` or `np.quantile` requests, use in-place operations (`out=image`, `+=`, `*=`) for large arrays in data loaders, and avoid redundant `.min()`/`.max()` scans immediately after a `clip` operation.
## 2024-05-25 - Tensor Augmentation Overhead
**Learning:** During continuous augmentation loops in PyTorch (e.g. data loaders), creating new tensors through arithmetic (`data + noise`) has a compounding performance and memory allocation cost.
**Action:** Use in-place operations (`data.add_(noise)`, `noise.mul_(sigma)`) whenever the original tensor does not need to be preserved. This reduces memory pressure on the allocator and slightly improves speed.
## 2024-06-01 - Pipeline Array Processing Optimization
**Learning:** In processing scripts such as DICOM inference pipelines, the same bottleneck with dense large array quantile operations and intermediate memory allocations is observed. Applying a stride to sample array values down before calling quantile calculations, followed by applying in-place arrays modifications, can result in significant (up to ~60% faster) array processing during batch processing.
**Action:** When updating normalization logic in dataset loaders, also verify if similar bottlenecks and code exist in the inference or bulk-processing loops. Reuse array slicing for percentiles/quantiles and prefer in-place arithmetic `+=`, `*=`, and `out=` where arrays are disposable.
## 2026-04-14 - Initialize DISTS calculator
**Learning:** Initializing Piq metrics (like DISTS) inside validation loops adds overhead or repeated instantiation logic which could be optimized by setting it in the class init method `__init__`.
**Action:** When a metric module is a Pytorch layer like `piq.DISTS()`, move it to `__init__` rather than initializing repeatedly during each step or batch of the validation.
