import plotly.express as px
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
    
def plot_loss_difference(log_probs, rep_str, seq_len):
    fig = px.line(
        utils.to_numpy(log_probs),
        hover_name = rep_str[1:],
        title = f"Per token log prob on correct token, for sequence of length {seq_len}",
        labels = {"index": "Sequence position", "value": "Log prob"}
    ).update_layout(showlegend = False, hovermode = "x unified")
    fig.add_vrect(x0=0, x1=seq_len - 0.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(
        x0=seq_len - 0.5, x1=2 * seq_len - 1, fillcolor="green", opacity=0.2, line_width=0
    )
    fig.show()