# tegen

Shared python code used by the Tegenfeldt group. Install by running

``` bash
pip install git+https://github.com/wduverger/tegen.git
```

Example usage

```python
import matplotlib.pyplot as plt
import tegen

# Read file
images = tegen.bioformats.read_file(filepath)

# Use image data
image = images['channel_name']

# Access image metadata
image.meta

# The following metadata is stored in image.meta, 
# but can also be accessed directly
image.pixel_size
image.origin_x
image.origin_y

# False-colour a polarisation stack
pol_rgb = tegen.polarisation.stack_to_rgb(image)
plt.imshow(pol_rgb)
```