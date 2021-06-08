# tegen

Shared python code used by the Tegenfeldt group. Install by running

``` bash
pip install git+https://github.com/wduverger/tegen.git
```

Example usage

```python
import matplotlib.pyplot as plt
import tegen
import cv2
import numpy as np

# Read file
images = tegen.bioformats.read_file('<file_path>')

# Use image data
image = images['<channel_name>']

# Access image metadata
image.meta

# The following metadata is stored in image.meta, 
# but can also be accessed directly
image.pixel_size
image.origin_x
image.origin_y

# False-colour a polarisation stack, supplying the angles at which the 
# excitation light was polarised at every frame in the pol_axis variable
pol_axis = np.arange(0, 170, 10)
pol_rgb = tegen.polarisation.stack_to_rgb(image, pol_axis)

# Show image in a plot with colourwheel and scalebar
ax = plt.imshow(pol_rgb)
tegen.polarisation.add_scalebar(ax, 10e-6/image.pixel_size, '10 μm')
tegen.polarisation.add_colourwheel(ax)

# Save image to file
pol_rgb_save = tegen.polarisation.add_scalebar_in_place(
    pol_rgb, 5, 2, 10e-6/image.pixel_size, '10 μm'
)
pol_rgb_save = tegen.polarisation.add_colourwheel_in_place(
    pol_rgb_save, 5, 40
)
cv2.imwrite('<file_path>', im*255)  # cv2 expects a range between 0-255, not 0-1

```