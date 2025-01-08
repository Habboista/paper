The chosen architecture was used for affine or relative MDE,
hence we use it for affine here (not relative as it is not easy to define a merging procedure).
--affine option must be present.
Under affine MDE the relevance of geometrical correction of perspective distorsions is secondary,
hence --preservecamera option is NOT used.
Also, we stick to the standard KITTI dataset for now by projecting raw data as did by Eigen et al.
Hence the options --velo and --project must be present.
The model is used in one of its standard form, i.e. w resnet50 encoder, hence --depth 50 is always used.
There is some instability on the affine loss function as it does not learn anything on random patches.
When using the scale invariant instead it learns something (at least it doesn't degenerate).
Solved it, so apparently the antialiasing option was working with skimage resize function and
it was causing very small depth values to appear in the ground truth that caused the numerical instability.
Also, this could have caused distorsions during the evaluation of the model and could explain the drop in accuracy as the prediction was scaled to match the ground truth size.
We'll see.
In case it would invalidate some of my intuitions.


EXP1, 2, 3:
preliminar comparison between sampling strategies


What happens when shifting $D$ to $[0, 1]$ before alignment?

$D_{[0,1]} = \frac{D - D_{min}}{D_{max} - D_{min}}$

$\hat{D} = \frac{D - t}{s}$


Normalization in [0, 1]; Trimming; Learning rate for backbone; aligning pred based on all the pixels (worse);
alignment of a different kind?