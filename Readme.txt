1.render_surface_points.py
render all surface points from any view, input val images, output all corresponding surface points.
only use sigma prediction
you can get surface points using depth or threshold(sigma>20)

2.render_visibility.py
using pretrained Nerf network to predict sigma, and sample on 400 directions to compute visibility
using visibility to compute transport map
using sh_util_gpu.py, the points in the same chunk share same directions

3.render_visibility_n.py
using pretrained Nerf network to predict sigma, and sample on 400 directions to compute visibility
using visibility to compute transport map
using sh_util_gpu_n.py, the points in the same chunk share DIFFERENT directions

4.render_visibility_n_predict.py
using pretained Nerf and Visibilty network, compute transport map using sample points and p_vis
compare the difference

5.train_visibility.py
using pretrained Nerf to train Visibility Network, using sample points to supervise
Only train visibility network

6.train_nert.py
use pretained Nerf, train albedo network, visibility network and Light.

7.sh_util_gpu_nert.py
used for nert training, using soft visibility map, so it is differencial