"""Flask web app (no Streamlit).  Run:  python flask_app.py
Then open http://127.0.0.1:5000 in any browser.
Uses pipeline.py for all the machine learning work."""
import io
import os
import threading
import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, abort, render_template_string, request, send_file

import pipeline as pl

app = Flask(__name__)

print("Loading / training models (first run downloads data from Kaggle)...")
bundle = pl.load_or_train(print)
raw = pd.read_csv(pl.CLEAN_PATH)
feature_cols = [c for c in raw.columns if c != pl.TARGET]
cats = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]

fields = []
for c in feature_cols:
    s = raw[c]
    if pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s)
        cast = int if is_int else float
        fields.append(dict(name=c, kind="num", min=cast(s.min()), max=cast(s.max()),
                           value=cast(s.median()), step=1 if is_int else "any"))
    else:
        fields.append(dict(name=c, kind="cat", options=sorted(s.dropna().unique().tolist())))

CHARTS = {
    "price": lambda: pl.fig_price_dist(raw),
    "corr": lambda: pl.fig_corr(raw),
    "actual": lambda: pl.fig_actual_pred(bundle),
    "errors": lambda: pl.fig_errors(bundle),
    "importance": lambda: pl.fig_importance(bundle),
    "elbow": lambda: pl.fig_elbow(bundle),
    "pca": lambda: pl.fig_pca(bundle),
}
for _c in cats:
    CHARTS["cat_" + _c] = (lambda c=_c: pl.fig_by_cat(raw, c))

last = {"df": None}

PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Smart House Price Predictor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#222}
 header{background:#1f3a5f;color:#fff;padding:18px 28px}
 header h1{margin:0;font-size:22px} header p{margin:4px 0 0;opacity:.8;font-size:13px}
 nav{display:flex;gap:6px;padding:12px 28px 0;flex-wrap:wrap}
 nav button{border:0;background:#dde4ee;padding:10px 16px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
 nav button.on{background:#fff;font-weight:600}
 .tab{display:none;background:#fff;margin:0 28px 28px;padding:22px;border-radius:0 8px 8px 8px}
 .tab.on{display:block}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
 label{font-size:13px;font-weight:600;display:block;margin-bottom:4px}
 input,select{width:100%;padding:8px;border:1px solid #bbb;border-radius:6px;box-sizing:border-box}
 .btn{background:#1f3a5f;color:#fff;border:0;padding:10px 20px;border-radius:6px;cursor:pointer;margin-top:16px;font-size:15px;text-decoration:none;display:inline-block}
 .cards{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}
 .card{background:#eef3fa;padding:14px 20px;border-radius:8px;min-width:150px}
 .card b{display:block;font-size:24px;margin-top:4px}
 .imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;margin-top:14px}
 .imgs img{max-width:100%;border:1px solid #e3e3e3;border-radius:6px}
 table{border-collapse:collapse;font-size:13px;margin:10px 0} th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}
 th{background:#eef3fa} .err{color:#b00020;font-weight:600}
 .wrap{overflow-x:auto}
</style></head><body>
<header><h1>🏠 Smart House Price Prediction &amp; Market Segmentation</h1>
<p>Regression · Random Forest · K-Means · Data Visualization</p></header>
<nav>
 {% for id,label in [('predict','🔮 Predict'),('batch','📄 Batch CSV'),('eda','📊 Data Explorer'),('model','🎯 Model Results'),('seg','🧩 Segments')] %}
 <button id="b-{{id}}" class="{{'on' if tab==id}}" onclick="show('{{id}}')">{{label}}</button>
 {% endfor %}
</nav>

<div id="t-predict" class="tab {{'on' if tab=='predict'}}">
 <h2>Enter house details</h2>
 <form method="post"><input type="hidden" name="mode" value="single">
 <div class="grid">
 {% for f in fields %}<div><label>{{f.name}}</label>
  {% if f.kind=='num' %}<input type="number" name="{{f.name}}" min="{{f.min}}" max="{{f.max}}" step="{{f.step}}" value="{{form.get(f.name, f.value)}}" required>
  {% else %}<select name="{{f.name}}">{% for o in f.options %}<option {{'selected' if form.get(f.name)==o}}>{{o}}</option>{% endfor %}</select>{% endif %}
 </div>{% endfor %}
 </div><button class="btn">Predict price</button></form>
 {% if error %}<p class="err">{{error}}</p>{% endif %}
 {% if result %}<div class="cards">
  <div class="card">Predicted price<b>{{result.price}}</b></div>
  <div class="card">Market segment<b>{{result.segment}}</b></div></div>{% endif %}
</div>

<div id="t-batch" class="tab {{'on' if tab=='batch'}}">
 <h2>Predict prices for many houses</h2>
 <p>Upload a CSV with columns: <code>{{feature_cols|join(', ')}}</code></p>
 <form method="post" enctype="multipart/form-data"><input type="hidden" name="mode" value="csv">
  <input type="file" name="file" accept=".csv" required style="max-width:340px">
  <br><button class="btn">Predict</button>
  <a class="btn" href="/template.csv">Download template</a></form>
 {% if batch_error %}<p class="err">{{batch_error}}</p>{% endif %}
 {% if table %}<a class="btn" href="/download.csv">Download predictions</a><div class="wrap">{{table|safe}}</div>{% endif %}
</div>

<div id="t-eda" class="tab">
 <h2>Exploratory data analysis</h2><p>{{n_rows}} houses after cleaning.</p>
 <div class="wrap">{{head|safe}}</div>
 <div class="imgs"><img src="/chart/price"><img src="/chart/corr">
 {% for c in cats %}<img src="/chart/cat_{{c}}">{% endfor %}</div>
</div>

<div id="t-model" class="tab">
 <h2>Model comparison (5-fold cross-validation)</h2>
 <div class="wrap">{{cv|safe}}</div>
 <p>Tuned Random Forest parameters: {{params}}</p>
 <div class="cards">
  <div class="card">Test MAE<b>{{m['MAE']}}</b></div>
  <div class="card">Test RMSE<b>{{m['RMSE']}}</b></div>
  <div class="card">Test R²<b>{{m['R²']}}</b></div></div>
 <div class="imgs"><img src="/chart/actual"><img src="/chart/errors"><img src="/chart/importance"></div>
</div>

<div id="t-seg" class="tab">
 <h2>Market segmentation (K-Means)</h2>
 <p>Chosen k = <b>{{k}}</b> (silhouette {{sil}}), segments named by average price.</p>
 <div class="wrap">{{profile|safe}}</div>
 <div class="imgs"><img src="/chart/elbow"><img src="/chart/pca"></div>
</div>

<script>
function show(id){
 document.querySelectorAll('.tab').forEach(e=>e.classList.remove('on'));
 document.querySelectorAll('nav button').forEach(e=>e.classList.remove('on'));
 document.getElementById('t-'+id).classList.add('on');
 document.getElementById('b-'+id).classList.add('on');
}
</script></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = dict(tab="predict", form={}, result=None, error=None, table=None, batch_error=None)
    if request.method == "POST":
        try:
            if request.form.get("mode") == "single":
                ctx["form"] = request.form.to_dict()
                row = {f["name"]: (float(request.form[f["name"]]) if f["kind"] == "num"
                                   else request.form[f["name"]]) for f in fields}
                res = pl.predict(bundle, pd.DataFrame([row])).iloc[0]
                ctx["result"] = dict(price=f"${res['Predicted Price']:,.0f}",
                                     segment=res["Market Segment"])
            else:
                ctx["tab"] = "batch"
                out = pl.predict(bundle, pd.read_csv(request.files["file"]))
                last["df"] = out
                ctx["table"] = out.head(200).to_html(index=False)
        except Exception as e:
            if request.form.get("mode") == "single":
                ctx["error"] = f"Could not predict: {e}"
            else:
                ctx["batch_error"] = f"Could not process file: {e}"
    m = bundle["metrics"]
    return render_template_string(
        PAGE, fields=fields, feature_cols=feature_cols, cats=cats, n_rows=f"{len(raw):,}",
        head=raw.head(15).to_html(index=False),
        cv=bundle["cv_table"].to_html(index=False, float_format=lambda x: f"{x:,.3f}"),
        params=bundle["best_params"],
        m={"MAE": f"{m['MAE']:,.0f}", "RMSE": f"{m['RMSE']:,.0f}", "R²": f"{m['R²']:.3f}"},
        k=bundle["best_k"], sil=f"{bundle['silhouette']:.3f}",
        profile=bundle["profile"].to_html(), **ctx)


@app.route("/chart/<name>")
def chart(name):
    if name not in CHARTS:
        abort(404)
    fig = CHARTS[name]()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/template.csv")
def template():
    buf = io.BytesIO(raw[feature_cols].head(5).to_csv(index=False).encode())
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="template.csv")


@app.route("/download.csv")
def download():
    if last["df"] is None:
        abort(404)
    buf = io.BytesIO(last["df"].to_csv(index=False).encode())
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")


def open_chrome():
    """Open the app in Google Chrome (falls back to the default browser)."""
    url = "http://127.0.0.1:5000"
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")):
        if os.path.exists(p):
            webbrowser.register("chrome", None, webbrowser.BackgroundBrowser(p))
            webbrowser.get("chrome").open(url)
            return
    webbrowser.open(url)


if __name__ == "__main__":
    print("\nReady! Opening http://127.0.0.1:5000 in Chrome...\n")
    threading.Timer(1.5, open_chrome).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)