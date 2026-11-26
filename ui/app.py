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
st.caption(
    "Builds on my Introduction to ML project, *Early Identification of At-Risk Learners* (OULAD). "
    "Same goal (find learners who need support before they fall behind), but the data is now "
    "**images of single handwritten characters** from children aged 6–10. "
    "Educational prototype: a screening aid, **not a diagnosis**."
)


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
    ("sensitivity_at_risk", "Sensitivity (at-risk caught)"),
    ("roc_auc", "ROC-AUC"),
]

health, health_err = api_get("/health")
info, _ = api_get("/model/info")

with st.sidebar:
    st.subheader("Model up-time")
    if health:
        st.metric("API status", "🟢 UP" if health["status"] == "ok" else "🔴 DOWN")
        st.metric("Uptime", fmt_uptime(health["uptime_seconds"]))
        st.caption(f"Started (UTC): {health['started_at'][:19]}")
        st.caption(f"Model file present: {health['model_loaded']}")
    else:
        st.error("API unreachable")
        st.caption(health_err)
    st.caption(f"API: `{API}`")

    if info:
        m = info.get("metrics") or {}
        if m:
            st.subheader("Current model (test set)")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
            c2.metric("F1 (macro)", f"{m.get('f1_macro', 0):.3f}")
        st.subheader("Retraining queue")
        pending, threshold = info.get("uploads_pending", 0), info.get("retrain_threshold", 20)
        st.progress(min(pending / max(threshold, 1), 1.0), text=f"{pending}/{threshold} new images")
        if info.get("retrain_recommended"):
            st.warning("Enough new data — retraining recommended.")
        st.caption(f"Auto-retrain: {'on' if info.get('auto_retrain_enabled') else 'off'}")
    if st.button("Refresh"):
        st.rerun()

tabs = st.tabs(
    ["🔍 Predict", "📊 Visualisations", "⬆️ Upload & retrain", "✅ Model evaluation", "⏱ Monitoring"]
)

with tabs[0]:
    st.markdown(
        "Upload a photo or scan of **one handwritten character** (letter or digit) by a child. "
        "The model returns the probability that the writing pattern matches the **dyslexic** group."
    )
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
                st.caption(f"API latency {result.get('latency_ms')} ms")
            with st.expander("Handwriting features computed for this image"):
                st.json(result["features"])
            st.info(
                "One character is weak evidence. In practice a teacher would screen many characters "
                "per child and refer for a proper assessment, never label a child from one image."
            )
        else:
            st.error(err)

with tabs[1]:
    data, err = api_get("/visualizations/features", timeout=180)
    if err:
        st.info("Feature table not available yet: " + str(err))
    else:
        df = pd.DataFrame(data["rows"])
        df["group"] = pd.Categorical(df["label_name"], categories=CLASS_LABELS, ordered=True)

        st.subheader("1 · Who is in the dataset?")
        c1, c2 = st.columns([1, 2])
        counts = df.groupby(["group", "split"], observed=True).size().reset_index(name="images")
        with c1:
            fig = px.bar(counts, x="group", y="images", color="split", barmode="stack", text="images",
                         title="Images per class and split", labels={"group": ""})
            fig.update_layout(height=340)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown(
                f"""
The data holds **{len(df)} single-character images** from **100 children aged 6–10** in schools in
Punjab, Pakistan: English letters, Urdu letters and digits. The two groups are **balanced by
design**, with half the images from children identified as dyslexic. The model therefore isn't
biased toward either answer.

A real classroom isn't balanced: roughly 5–10 % of children have dyslexia. Accuracy on
balanced data overstates how useful a screener is, so we also report **sensitivity** (how many
at-risk children we catch) and **specificity** (how many we correctly leave alone).
"""
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
**What each feature means**

- **Ink coverage**: the share of the image that is pencil stroke, i.e. how large and heavy the
  letter is. Letters from the dyslexic group cover **slightly more** of the frame. That fits
  larger, less controlled letter forms and overwriting.
- **Stroke darkness**: the contrast between pencil and paper, a proxy for pen pressure. The
  dyslexic group writes **a little lighter** (median 0.65 vs 0.70). Uneven or light pressure is
  a known feature of effortful handwriting.
- **Stroke thinness**: stroke-edge pixels per ink pixel. Thin, single, confident strokes score
  high; thick, retraced or gone-over strokes score low. The dyslexic group scores **lower**,
  which fits children going back over letters they are unsure of.

**The story:** every difference points the same way, toward *effortful, less automatic writing*.
But the boxes **overlap almost completely**: on its own, each feature separates the groups only a
little better than chance (AUC 0.47–0.54). Dyslexia doesn't show up as a simple ink statistic.
It shows up in the **shape** of letters: reversals (b/d, p/q, mirrored digits), malformed
strokes, odd proportions. That is why a CNN that looks at letter shape is needed, and why it
does much better than these features alone.
"""
        )

        st.subheader("3 · Feature space")
        fig = px.scatter(
            df, x="stroke_darkness", y="stroke_thinness", color="group", color_discrete_map=COLORS,
            opacity=0.55, title="Pen pressure vs stroke thinness, coloured by group",
            labels={"stroke_darkness": "Stroke darkness", "stroke_thinness": "Stroke thinness", "group": ""},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
        st.caption("The two clouds sit on top of each other: hand-crafted features alone cannot screen reliably.")

with tabs[2]:
    st.markdown(
        "**Step 1: upload bulk labelled data.** Either a **zip** with one folder per class "
        "(`dyslexic` / `non_dyslexic`, or `Yes` / `No` as in the original dataset), or **many images** "
        "that belong to one class."
    )
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

    st.markdown(
        "**Saved in the database.** Each uploaded image is preprocessed (128×128 grayscale) and stored "
        "in SQLite (`data/pipeline.db`) as **pending**. Retraining reads the pending images from the "
        "database and marks them **used** once the retrained model goes live."
    )
    db_data, db_err = api_get("/data/uploads")
    if db_data:
        summ = db_data["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images in database", summ["total"])
        m2.metric("Pending (not yet trained on)", summ["pending"])
        m3.metric("Used by a retrained model", summ["used"])
        m4.metric("Retraining runs", summ["retrain_runs"])
        if db_data["uploads"]:
            st.dataframe(
                pd.DataFrame(db_data["uploads"])[
                    ["id", "original_name", "class_label", "source", "width", "height", "png_bytes",
                     "uploaded_at", "status", "retrain_run_id"]
                ],
                hide_index=True,
                height=240,
            )
        else:
            st.caption("No uploads yet.")
        runs, _ = api_get("/data/retrain-runs")
        if runs and runs["runs"]:
            with st.expander("Retraining runs (from the database)"):
                st.dataframe(pd.DataFrame(runs["runs"]), hide_index=True)
    else:
        st.caption(f"Database not reachable: {db_err}")

    st.divider()
    st.markdown(
        "**Step 2: retrain.** The current model is **fine-tuned** on all training data, including your "
        "uploads. Before going live it must pass a **quality gate**: it replaces the current model only "
        "if its macro F1 on the held-out test set doesn't drop by more than 0.02."
    )
    epochs = st.slider("Retraining epochs", 1, 15, 6)
    if st.button("🚀 Trigger retraining", type="primary"):
        body, err = api_post("/retrain", params={"epochs": epochs}, timeout=30)
        st.json(body or {"error": err})

    status, _ = api_get("/retrain/status")
    if status:
        state = status.get("state")
        box = {"running": st.info, "complete": st.success, "error": st.error}.get(state, st.write)
        box(f"**Retrain job: {state}** — {status.get('message', '')}")
        if state == "complete" and status.get("previous_f1_macro") is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("F1 macro (new)", f"{status['f1_macro']:.3f}",
                      f"{status['f1_macro'] - status['previous_f1_macro']:+.3f}")
            c2.metric("Accuracy (new)", f"{status['accuracy']:.3f}",
                      f"{status['accuracy'] - status['previous_accuracy']:+.3f}")
            c3.metric("Promoted", "yes" if status.get("promoted") else "no (kept old model)")
        if state == "running":
            st.caption("Training runs in the background. This page refreshes every 5 s.")
            time.sleep(5)
            st.rerun()

with tabs[3]:
    st.markdown("Metrics recorded when the current model was trained, and a **live evaluation** run inside the deployed API.")
    if info and info.get("metrics"):
        m = info["metrics"]
        cols = st.columns(len(METRIC_LABELS))
        for col, (k, label) in zip(cols, METRIC_LABELS):
            col.metric(label, f"{m[k]:.3f}" if m.get(k) is not None else "–")
        c1, c2 = st.columns(2)
        with c1:
            if info.get("confusion_matrix"):
                st.plotly_chart(confusion_fig(info["confusion_matrix"], "Confusion matrix at training time"),
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
    if st.button("▶ Evaluate the live model now (in production)", type="primary"):
        with st.spinner("Running the deployed model on the held-out test images…"):
            ev, err = api_post("/evaluate", timeout=300)
        if ev:
            st.success(f"Evaluated {ev['n_test']} images in {ev['eval_seconds']} s at {ev['evaluated_at'][:19]} UTC")
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
        c3.metric("Mean latency (ms)", lat.get("mean_ms") or "–")
        c4.metric("p95 latency (ms)", lat.get("p95_ms") or "–")
        st.caption(
            "Server-side /predict latency over the last 500 requests in the replica that answered. "
            "Locust flood-test results are in the README."
        )
    else:
        st.error(err)
    if info:
        st.markdown("**Images on disk**")
        st.dataframe(pd.DataFrame({"train": info["train_counts"], "test": info["test_counts"]}),
                     width="stretch")
