"""Streamlit UI: up-time, prediction, visualisations, upload + retrain, evaluation."""

import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
CLASS_LABELS = ["Non-dyslexic", "Dyslexic (at risk)"]
COLORS = {"Non-dyslexic": "#2a9d8f", "Dyslexic (at risk)": "#e76f51"}
FEATURES = {
    "ink_coverage": "Ink coverage",
    "stroke_darkness": "Stroke darkness (pen pressure)",
    "stroke_thinness": "Stroke thinness",
}

st.set_page_config(page_title="Dyslexia Screening Pipeline", page_icon="✏️", layout="wide")
st.title("Early dyslexia screening from children's handwriting")
st.caption("Screens one handwritten character for signs of dyslexia. A screening aid, **not a diagnosis**.")


def api_get(path: str, timeout: int = 30):
    try:
        r = requests.get(f"{API}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, timeout: int = 120, **kwargs):
    try:
        r = requests.post(f"{API}{path}", timeout=timeout, **kwargs)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"text": r.text}
        return (body, None) if r.ok else (None, body.get("detail", body))
    except Exception as exc:
        return None, str(exc)


def fmt_uptime(secs: float) -> str:
    days, rem = divmod(int(secs), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return (f"{days}d " if days else "") + f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def confusion_fig(cm, title: str):
    fig = px.imshow(
        cm, x=CLASS_LABELS, y=CLASS_LABELS, text_auto=True, color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "True", "color": "Images"}, title=title,
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


METRIC_LABELS = [
    ("accuracy", "Accuracy"),
    ("precision_macro", "Precision"),
    ("recall_macro", "Recall"),
    ("f1_macro", "F1 (macro)"),
    ("sensitivity_at_risk", "Sensitivity"),
    ("roc_auc", "ROC-AUC"),
]

with st.spinner("Connecting to the API (up to a minute if it was asleep)…"):
    health, health_err = api_get("/health", timeout=90)
info, _ = api_get("/model/info")

with st.sidebar:
    st.subheader("Model up-time")
    if health:
        st.metric("API status", "🟢 UP" if health["status"] == "ok" else "🔴 DOWN")
        st.metric("Uptime", fmt_uptime(health["uptime_seconds"]))
        st.caption(f"Started (UTC): {health['started_at'][:19]}")
    else:
        st.error("API unreachable. Wait a moment and press Refresh.")

    if info:
        m = info.get("metrics") or {}
        if m:
            st.subheader("Current model")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
            c2.metric("F1 (macro)", f"{m.get('f1_macro', 0):.3f}")
        st.subheader("New images")
        pending, threshold = info.get("uploads_pending", 0), info.get("retrain_threshold", 20)
        st.progress(min(pending / max(threshold, 1), 1.0), text=f"{pending}/{threshold} before retraining")
        if info.get("retrain_recommended"):
            st.warning("Retraining recommended.")
    if st.button("Refresh"):
        st.rerun()

tabs = st.tabs(
    ["🔍 Predict", "📊 Visualisations", "⬆️ Upload & retrain", "✅ Model evaluation", "⏱ Monitoring"]
)

with tabs[0]:
    st.markdown("Upload an image of **one handwritten character**.")
    uploaded = st.file_uploader("Handwritten character image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if uploaded and st.button("Run prediction", type="primary"):
        with st.spinner("Scoring image…"):
            result, err = api_post(
                "/predict", timeout=60,
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/png")},
            )
        if result:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(uploaded, caption=uploaded.name, width=240)
            with c2:
                risk = result["at_risk_probability"]
                box = st.error if result["class_id"] == 1 else st.success
                box(f"**{result['label']}** — confidence {result['confidence']:.1%}")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=risk * 100, number={"suffix": "%"},
                    title={"text": "Probability of at-risk pattern"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#e76f51"},
                           "steps": [{"range": [0, 50], "color": "#e9f5f3"}, {"range": [50, 100], "color": "#fbe9e4"}]},
                ))
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Response time: {result.get('latency_ms')} ms")
            with st.expander("Handwriting features"):
                st.json(result["features"])
        else:
            st.error(err)

with tabs[1]:
    data, err = api_get("/visualizations/features", timeout=180)
    if err:
        st.info("Feature table not available yet: " + str(err))
    else:
        df = pd.DataFrame(data["rows"])
        df["group"] = pd.Categorical(df["label_name"], categories=CLASS_LABELS, ordered=True)

        st.subheader("1 · The dataset")
        c1, c2 = st.columns([1, 2])
        counts = df.groupby(["group", "split"], observed=True).size().reset_index(name="images")
        with c1:
            fig = px.bar(counts, x="group", y="images", color="split", barmode="stack", text="images",
                         title="Images per class and split", labels={"group": ""})
            fig.update_layout(height=340)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown(
                f"**{len(df)} images** of single characters from **100 children aged 6–10**. "
                "The two classes are **balanced**, 50/50."
            )

        st.subheader("2 · Three handwriting features")
        means = df.groupby("group", observed=True)[list(FEATURES)].median()
        cols = st.columns(3)
        for col, (key, name) in zip(cols, FEATURES.items()):
            with col:
                fig = px.box(df, x="group", y=key, color="group", color_discrete_map=COLORS,
                             points=False, title=name, labels={key: name, "group": ""})
                fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=40, b=10))
                if key == "ink_coverage":
                    fig.update_yaxes(type="log")
                st.plotly_chart(fig, width="stretch")
                non, dys = means[key].iloc[0], means[key].iloc[1]
                st.metric("Median: dyslexic vs non-dyslexic", f"{dys:.3f}", f"{(dys - non) / non * 100:+.1f}%",
                          delta_color="off")

        st.markdown(
            """
- **Ink coverage:** dyslexic group writes slightly **bigger, heavier** letters.
- **Stroke darkness** (pen pressure): dyslexic group writes slightly **lighter**.
- **Stroke thinness:** dyslexic group has **thicker, retraced** strokes.

**Story:** all three point to *effortful writing*, but the groups overlap a lot. Dyslexia shows in
letter **shape**, which is why the CNN works much better than these features.
"""
        )

        st.subheader("3 · Feature space")
        fig = px.scatter(
            df, x="stroke_darkness", y="stroke_thinness", color="group", color_discrete_map=COLORS,
            opacity=0.55, title="Pen pressure vs stroke thinness",
            labels={"stroke_darkness": "Stroke darkness", "stroke_thinness": "Stroke thinness", "group": ""},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
        st.caption("The groups overlap, so these features alone can't screen reliably.")

with tabs[2]:
    st.markdown("**Step 1 · Upload data:** a zip with `Yes` / `No` folders, or several images of one class.")
    c1, c2 = st.columns(2)
    with c1:
        zip_file = st.file_uploader("Zip of class folders", type=["zip"])
    with c2:
        loose = st.file_uploader(
            "Or several images of one class", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )
        class_id = st.radio("Class for these images", options=[0, 1], format_func=lambda i: CLASS_LABELS[i],
                            horizontal=True)
    if st.button("Upload data"):
        files, form = [], {}
        if zip_file:
            files.append(("archive", (zip_file.name, zip_file.getvalue(), "application/zip")))
        if loose:
            files += [("files", (f.name, f.getvalue(), f.type or "image/png")) for f in loose]
            form["class_id"] = str(class_id)
        if not files:
            st.warning("Choose a zip or images first.")
        else:
            with st.spinner("Uploading and preprocessing…"):
                body, err = api_post("/upload-data", files=files, data=form, timeout=300)
            if body:
                st.success(f"Saved {body['saved']} images (skipped {body.get('skipped', 0)}). {body['message']}")
            else:
                st.error(err)

    st.markdown("**Database** · uploads are saved as **pending**, then marked **used** after retraining.")
    db_data, db_err = api_get("/data/uploads")
    if db_data:
        summ = db_data["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images", summ["total"])
        m2.metric("Pending", summ["pending"])
        m3.metric("Used", summ["used"])
        m4.metric("Retraining runs", summ["retrain_runs"])
        if db_data["uploads"]:
            st.dataframe(
                pd.DataFrame(db_data["uploads"])[
                    ["id", "original_name", "class_label", "source", "uploaded_at", "status", "retrain_run_id"]
                ],
                hide_index=True,
                height=240,
            )
        else:
            st.caption("No uploads yet.")
        runs, _ = api_get("/data/retrain-runs")
        if runs and runs["runs"]:
            with st.expander("Retraining runs"):
                st.dataframe(pd.DataFrame(runs["runs"]), hide_index=True)
    else:
        st.caption(f"Database not reachable: {db_err}")

    st.divider()
    st.markdown(
        "**Step 2 · Retrain:** fine-tunes the current model. The new model is kept only if its F1 "
        "doesn't drop by more than 0.02."
    )
    epochs = st.slider("Epochs", 1, 10, 2)
    if st.button("🚀 Trigger retraining", type="primary"):
        body, err = api_post("/retrain", params={"epochs": epochs}, timeout=30)
        st.json(body or {"error": err})

    status, _ = api_get("/retrain/status")
    if status:
        state = status.get("state")
        box = {"running": st.info, "complete": st.success, "error": st.error}.get(state, st.write)
        box(f"**Retraining: {state}**. {status.get('message', '')}")
        if state == "complete" and status.get("previous_f1_macro") is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("F1 (new)", f"{status['f1_macro']:.3f}",
                      f"{status['f1_macro'] - status['previous_f1_macro']:+.3f}")
            c2.metric("Accuracy (new)", f"{status['accuracy']:.3f}",
                      f"{status['accuracy'] - status['previous_accuracy']:+.3f}")
            c3.metric("New model kept", "yes" if status.get("promoted") else "no")
        if state == "running":
            st.caption("Refreshing every 5 s…")
            time.sleep(5)
            st.rerun()

with tabs[3]:
    if info and info.get("metrics"):
        m = info["metrics"]
        cols = st.columns(len(METRIC_LABELS))
        for col, (k, label) in zip(cols, METRIC_LABELS):
            col.metric(label, f"{m[k]:.3f}" if m.get(k) is not None else "–")
        c1, c2 = st.columns(2)
        with c1:
            if info.get("confusion_matrix"):
                st.plotly_chart(confusion_fig(info["confusion_matrix"], "Confusion matrix"),
                                width="stretch")
        hist, _ = api_get("/model/history")
        with c2:
            if hist and hist.get("history"):
                h = pd.DataFrame(hist["history"])
                h["epoch"] = range(1, len(h) + 1)
                fig = go.Figure()
                for k, dash in [("loss", None), ("val_loss", "dash")]:
                    fig.add_scatter(x=h["epoch"], y=h[k], name=k, line=dict(dash=dash))
                for k, dash in [("accuracy", None), ("val_accuracy", "dash")]:
                    fig.add_scatter(x=h["epoch"], y=h[k], name=k, yaxis="y2", line=dict(dash=dash))
                fig.update_layout(title=f"Training curves ({hist.get('trained_mode', '')})", height=380,
                                  yaxis=dict(title="loss"),
                                  yaxis2=dict(title="accuracy", overlaying="y", side="right", range=[0, 1]))
                st.plotly_chart(fig, width="stretch")

    st.divider()
    if st.button("▶ Evaluate the live model", type="primary"):
        with st.spinner("Evaluating on the test images…"):
            ev, err = api_post("/evaluate", timeout=300)
        if ev:
            st.success(f"Evaluated {ev['n_test']} test images in {ev['eval_seconds']} s")
            cols = st.columns(len(METRIC_LABELS))
            for col, (k, label) in zip(cols, METRIC_LABELS):
                col.metric(label, f"{ev[k]:.3f}" if ev.get(k) is not None else "–")
            st.plotly_chart(confusion_fig(ev["confusion_matrix"], "Confusion matrix (live evaluation)"),
                            width="stretch")
            st.dataframe(pd.DataFrame(ev["classification_report"]).T.round(3), width="stretch")
        else:
            st.error(err)

with tabs[4]:
    lat, err = api_get("/production/latency")
    if lat:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Uptime", fmt_uptime(lat.get("uptime_seconds") or (health or {}).get("uptime_seconds", 0)))
        c2.metric("Predictions served", lat.get("n"))
        c3.metric("Mean response (ms)", lat.get("mean_ms") or "–")
        c4.metric("p95 response (ms)", lat.get("p95_ms") or "–")
    else:
        st.error(err)
    if info:
        st.markdown("**Images**")
        st.dataframe(pd.DataFrame({"train": info["train_counts"], "test": info["test_counts"]}),
                     width="stretch")
