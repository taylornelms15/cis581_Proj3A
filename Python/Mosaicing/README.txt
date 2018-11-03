How to Run:

You can run this project by executing ''python mymosaic.py'' from the command line with options that specify the images to be stitched. To do a nearest neighbors calculation, we used the KDTree from scipy.spatial.

	We use the ''argparse'' module to handle various inputs. These are the available options:

	-l: Specify the path to the left image.
	-m: Specify the path to the middle image.
	-r: Specify the path to the right image.


Please note that this project works for stitching strictly three images together.

Potential Packages to Install:
	- argparse
	- scipy
