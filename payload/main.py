import jax

# The total number of TPU cores in the Pod
device_count = jax.device_count()

# The number of TPU cores attached to this host
local_device_count = jax.local_device_count()

# The psum is performed over all mapped devices across the Pod
xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)





# Write results to a file from a single host to avoid duplicated output
import os
dest_dir = os.getenv('DEST_DIR', './')

with open(os.path.join(dest_dir, 'results.txt'), 'w') as f:
    f.write(f'global device count: {device_count}\n')
    f.write(f'local device count: {local_device_count}\n')
    f.write(f'pmap result: {r}\n')
    f.write(f'process index: {jax.process_index()}\n')