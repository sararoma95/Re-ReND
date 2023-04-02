# Viewer
The viewer code is provided in this repo, as four <code>.html</code> files for two types of datasets.
You can set up a local server on your machine, e.g.,
```
cd folder_containing_the_html
python -m http.server -p 8080
```
Then open
```
localhost:8000/free_syn.html?obj=chair&qs6&ntex8
```
Note that you should put the *UVWB textures*, *minmax.json* and *.obj* inside of <code>obj_name + "/meshes_textures_" + tri_size + "_" + num_textures</code> folder. The folder should be in the same directory as the html file.

<code>num_textures</code> is <code>8</code> if you use embedding dimesion = 32 (<code>args.components</code>) or <code>16</code> in case you use 64.

Please allow some time for the scenes to load. Use left mouse button to rotate, right mouse button to pan, and scroll wheel to zoom. On phones, Use you fingers to rotate or pan or zoom. Resize the window (or landscape<->portrait your phone) to show the resolution.

This part of the implementation is based on MobileNeRF's github.