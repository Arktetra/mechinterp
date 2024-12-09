import plotly as px
import transformer_lens.utils as utils

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint = 0.0,
        color_continuous_scale = "RdBu",
        **kwargs,
    ).show()
    
def line(tensor, **kwargs):
    px.line(
        utils.to_numpy(tensor),
        **kwargs,
    ).show()
    
def scatter(x, y, xaxis = "", yaxis = "", caxis = "", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y = y,
        x = x,
        labels = {"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs
    ).show()