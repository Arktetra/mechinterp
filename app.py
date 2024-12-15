from flask import Flask, redirect, render_template, request
from jinja2 import Template

from mechinterp.heads import plot_attn_pattern, induction_attn_detector
from mechinterp.transformer_utils import get_attn_only_2L_transformer
from mechinterp.utils import run_and_cache_model_repeated_tokens


model = get_attn_only_2L_transformer(
    from_pretrained = True
)
rep_tokens, rep_logits, rep_cache = run_and_cache_model_repeated_tokens(
    model,
    seq_len = 25
)

app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/cirvis")

@app.route("/cirvis")
def cirvis():
    return render_template("index.html")

@app.route("/attention")
def attention():
    layers = model.cfg.n_layers
    heads = model.cfg.n_heads
    return render_template("attention.html", layers = layers, heads = heads)

@app.route("/induction")
def induction():
    induction_heads = induction_attn_detector(
        model = model, cache = rep_cache
    )
    
    return render_template("induction.html", induction_heads = induction_heads)

@app.route("/pattern", methods = ["POST"])
def show_pattern():
    layer = int(request.values.get("layer"))
    head = int(request.values.get("head"))
    title = request.values.get("title")
    title = f"L{layer}H{head} " + title
    
    fig_html = plot_attn_pattern(
        cache = rep_cache,
        layer = layer,
        head_idx = head,
        tokens = model.to_str_tokens(rep_tokens),
        return_type = "html",
        labels = {"x": "Source", "y": "Destination"},
        title = title
    )
    
    plotly_jinja_data = {"fig": fig_html}
    
    with open("./templates/pattern.html") as template_file:
        template = Template(template_file.read())
        return template.render(plotly_jinja_data)