This experiment was for comparing warp vs crop during patch sampling,
but, as it was performed using a random patch sampling and affine depth,
it proved that the RANDOM sampling with AFFINE leads to a local minimum,
nothing is learned as it can be seen in visualize_model_on_patches.

Consider a predicted disparity map $D$ with $D(x,y) \approx \mathcal{N}(0, 1)$
and a ground truth depth map $Z^{*}$.
When applying random sampling, many patches contain planar regions.
Planar regions in space correspond to regions defined by $D^{*}(x,y) = ax + by + c$
in disparity space.
Hence, after alignment, $D$ doesn't change much as it is already aligned.
$\mathbb{E}\{D^{*}\} = \frac{aw}{2} + \frac{bh}{2} + c = \frac{(a + b)h}{2} + c$,
$\mathbb{Var}\{D^{*}\} = \frac{(aw)^{2}}{12} + \frac{(bh)^{2}}{12} = \frac{(a + b)^{2}h^{2}}{12}$
If $D^{*}$ is aligned, we get $D^{*}(x, y) \approx 0$.