import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from PIL import Image
    import matplotlib.pyplot as plt
    import OpenEXR

    return Image, OpenEXR, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example visualization of .png depth image
    Note that all images have same scale for depth
    """)
    return


@app.cell
def _(Image, plt):

    _img = Image.open("output/Example/0000_rgb.png")
    _depth_img = Image.open("output/Example/0000_depth.png")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(_img, cmap='gray')
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(_depth_img)
    plt.axis("off")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example usage of the .exr file
    required inverted normalization of the depth as unlike in natural scenes we want the depth to be computed from the background towards the camera, such that the depth at the background is at 0.
    """)
    return


@app.cell
def _(OpenEXR):
    import numpy as np
    exrfile = OpenEXR.File("output/Prorocentrum micans/0000_depth.exr")

    R = exrfile.channels()["RGBA"].pixels
    return R, np


@app.cell
def _(R, np, plt):
    depth = R[:, :, 0].astype(np.float32).copy()

    # Identify object pixels
    mask = np.isfinite(depth) & (depth < 1.0)

    # Normalize only object depth values
    d_min = depth[mask].min()
    d_max = depth[mask].max()

    normalized = np.zeros_like(depth)

    # Inverted normalization
    normalized[mask] = (d_max - depth[mask]) / (d_max - d_min)

    # Visualize (closer = brighter)
    plt.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Normalized depth")
    plt.axis("off")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
