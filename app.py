from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from depth_capture.config import DepthCaptureConfig, default_model_id
from depth_capture.engine import DepthAnythingEngine
from depth_capture.history import JobRecord, JobStore
from depth_capture.i18n import t
from depth_capture.pipeline import DepthPipeline, ProgressInfo, inspect_video, unique_path

INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
HISTORY_PATH = OUTPUT_DIR / "jobs.json"

DEFAULT_PARAMS = {
    "scale": 0.5,
    "stride": 1,
    "gamma": 0.7,
    "crush": 35.0,
    "smooth": 0.4,
    "preserve_audio": True,
}

STYLES = """
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap");

:root {
  --bg: #0b0d10;
  --bg-elev: #12171f;
  --bg-panel: #10141b;
  --line: #243044;
  --text: #e8edf2;
  --muted: #8b9bb0;
  --accent: #3d7ea6;
  --monitor: #07090c;
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

.stApp {
  background: var(--bg);
  color: var(--text);
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"] {
  display: none !important;
}

.block-container {
  padding: 0.7rem 1rem 1.2rem 1rem !important;
  max-width: 100% !important;
}

.chrome {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 8px 4px 12px 4px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--line);
  position: sticky;
  top: 0;
  z-index: 20;
  background: color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter: blur(8px);
}
.chrome-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.logo {
  width: 32px;
  height: 32px;
  border: 1px solid var(--line);
  background: var(--bg-elev);
  color: var(--text);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  border-radius: 6px;
  flex-shrink: 0;
}
.chrome-copy strong {
  display: block;
  font-size: 0.92rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
}
.chrome-copy span {
  color: var(--muted);
  font-size: 0.75rem;
}

.panel-label {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 8px 0;
}

.monitor-empty {
  border: 1px dashed var(--line);
  background: var(--monitor);
  border-radius: 10px;
  min-height: 220px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  text-align: center;
  padding: 20px;
}
.monitor-empty strong {
  display: block;
  color: var(--text);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: 0.16em;
  font-size: 0.78rem;
  margin-bottom: 8px;
}
.monitor-empty p { margin: 0; font-size: 0.82rem; }

.monitor-wrap { position: relative; }
.meta-overlay {
  position: absolute;
  right: 10px;
  bottom: 36px;
  z-index: 3;
  background: rgba(0, 0, 0, 0.62);
  color: #f3f6fa;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10px;
  letter-spacing: 0.04em;
  padding: 4px 7px;
  border-radius: 4px;
  pointer-events: none;
}

[data-testid="stVideo"] {
  background: var(--monitor);
  border: 1px solid var(--line);
  border-radius: 10px;
  overflow: hidden;
}
[data-testid="stVideo"] video,
[data-testid="stImage"] img {
  max-height: 280px !important;
  object-fit: contain !important;
  background: var(--monitor);
}

.timeline {
  margin-top: 14px;
  padding: 10px 12px 14px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--bg-panel);
}
.timeline-label {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.timeline-track {
  height: 28px;
  border-radius: 6px;
  background:
    repeating-linear-gradient(90deg, #1c2430 0 10px, #151b24 10px 20px);
  position: relative;
  overflow: hidden;
}
.timeline-block {
  position: absolute;
  left: 8%;
  width: 54%;
  top: 5px;
  height: 18px;
  border-radius: 4px;
  background: #2a3a4e;
  opacity: 0.85;
}

.history-empty {
  color: var(--muted);
  font-size: 0.82rem;
  padding: 12px 4px;
}

.stButton > button {
  border-radius: 8px;
  font-weight: 600;
}
</style>
"""

LIGHT_OVERRIDES = """
<style>
:root {
  --bg: #f4f6f8;
  --bg-elev: #ffffff;
  --bg-panel: #eef1f5;
  --line: #d5dce6;
  --text: #1b2430;
  --muted: #5d6b7c;
  --accent: #2f6f98;
  --monitor: #10141a;
}
</style>
"""

SYSTEM_OVERRIDES = """
<style>
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f4f6f8;
    --bg-elev: #ffffff;
    --bg-panel: #eef1f5;
    --line: #d5dce6;
    --text: #1b2430;
    --muted: #5d6b7c;
    --monitor: #10141a;
  }
}
</style>
"""


def _store() -> JobStore:
    return JobStore(HISTORY_PATH)


def _save_upload(uploaded) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    target = INPUT_DIR / uploaded.name
    if target.exists():
        target = unique_path(INPUT_DIR, Path(uploaded.name).stem, suffix)
    target.write_bytes(uploaded.getbuffer())
    return target


def _format_eta(seconds: float) -> str:
    total = int(max(seconds, 0))
    minutes, secs = divmod(total, 60)
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _init_state() -> None:
    st.session_state.setdefault("lang", "zh")
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("show_params", False)
    st.session_state.setdefault("current_job_id", None)
    st.session_state.setdefault("draft_source", None)
    st.session_state.setdefault("draft_upload_name", None)
    for key, value in DEFAULT_PARAMS.items():
        st.session_state.setdefault(f"p_{key}", value)


def _apply_params(params: dict) -> None:
    st.session_state.p_scale = float(params.get("scale", DEFAULT_PARAMS["scale"]))
    st.session_state.p_stride = int(params.get("stride", DEFAULT_PARAMS["stride"]))
    st.session_state.p_gamma = float(params.get("gamma", DEFAULT_PARAMS["gamma"]))
    st.session_state.p_crush = float(params.get("crush", DEFAULT_PARAMS["crush"]))
    st.session_state.p_smooth = float(params.get("smooth", DEFAULT_PARAMS["smooth"]))
    st.session_state.p_preserve_audio = bool(params.get("preserve_audio", True))


def _current_params() -> dict:
    return {
        "scale": float(st.session_state.p_scale),
        "stride": int(st.session_state.p_stride),
        "gamma": float(st.session_state.p_gamma),
        "crush": float(st.session_state.p_crush),
        "smooth": float(st.session_state.p_smooth),
        "preserve_audio": bool(st.session_state.p_preserve_audio),
    }


@st.cache_resource(show_spinner=False)
def _load_engine(model_id: str) -> DepthAnythingEngine:
    return DepthAnythingEngine.load(model_id)


def _inject_styles(theme: str) -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    if theme == "light":
        st.markdown(LIGHT_OVERRIDES, unsafe_allow_html=True)
    elif theme == "system":
        st.markdown(SYSTEM_OVERRIDES, unsafe_allow_html=True)


def _render_header(lang: str) -> None:
    left, right = st.columns([4.2, 1.3])
    with left:
        st.markdown(
            f"""
            <div class="chrome" style="border:0;margin:0;padding:4px 0;background:transparent">
              <div class="chrome-brand">
                <div class="logo">DC</div>
                <div class="chrome-copy">
                  <strong>{t(lang, "app_title")}</strong>
                  <span>{t(lang, "app_desc")}</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        lang_col, more_col = st.columns([1.2, 0.8])
        with lang_col:
            next_lang = "en" if lang == "zh" else "zh"
            label = t(lang, "lang_en") if lang == "zh" else t(lang, "lang_zh")
            if st.button(label, key="lang_toggle", use_container_width=True):
                st.session_state.lang = next_lang
                st.rerun()
        with more_col:
            with st.popover(t(lang, "more")):
                st.caption(t(lang, "theme"))
                choice = st.radio(
                    t(lang, "theme"),
                    options=["system", "light", "dark"],
                    index=["system", "light", "dark"].index(st.session_state.theme),
                    format_func=lambda value: {
                        "system": t(lang, "theme_system"),
                        "light": t(lang, "theme_light"),
                        "dark": t(lang, "theme_dark"),
                    }[value],
                    label_visibility="collapsed",
                    key="theme_radio",
                )
                if choice != st.session_state.theme:
                    st.session_state.theme = choice
                    st.rerun()


def _render_history(lang: str, store: JobStore, jobs: list[JobRecord]) -> None:
    current = st.session_state.current_job_id
    st.markdown(f'<div class="panel-label">{t(lang, "history")}</div>', unsafe_allow_html=True)
    if st.button(
        t(lang, "new"),
        key="new_job",
        use_container_width=True,
        type="primary" if current is None else "secondary",
    ):
        st.session_state.current_job_id = None
        st.session_state.draft_source = None
        st.session_state.draft_upload_name = None
        _apply_params(DEFAULT_PARAMS)
        st.rerun()

    if not jobs:
        st.markdown(
            f'<div class="history-empty">{t(lang, "empty_history")}</div>',
            unsafe_allow_html=True,
        )
        return

    for job in jobs:
        selected = job.id == current
        row = st.columns([4.2, 1])
        with row[0]:
            title = job.title
            if st.button(
                title,
                key=f"sel_{job.id}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state.current_job_id = job.id
                _apply_params(job.params or DEFAULT_PARAMS)
                st.rerun()
        with row[1]:
            if st.button("✕", key=f"del_{job.id}", help=t(lang, "delete")):
                store.delete(job.id)
                if st.session_state.current_job_id == job.id:
                    st.session_state.current_job_id = None
                    _apply_params(DEFAULT_PARAMS)
                st.rerun()
        stamp = job.created_at.replace("T", " ")[:16]
        st.caption(stamp)


def _render_params(lang: str, output_path: Path | None) -> None:
    st.markdown(f'<div class="panel-label">{t(lang, "params")}</div>', unsafe_allow_html=True)
    st.slider(t(lang, "scale"), min_value=0.25, max_value=1.0, step=0.05, key="p_scale", help=t(lang, "scale_help"))
    st.slider(t(lang, "stride"), min_value=1, max_value=8, step=1, key="p_stride", help=t(lang, "stride_help"))
    st.slider(t(lang, "gamma"), min_value=0.1, max_value=2.0, step=0.05, key="p_gamma", help=t(lang, "gamma_help"))
    st.slider(t(lang, "crush"), min_value=0.0, max_value=80.0, step=1.0, key="p_crush", help=t(lang, "crush_help"))
    st.slider(t(lang, "smooth"), min_value=0.0, max_value=0.9, step=0.05, key="p_smooth", help=t(lang, "smooth_help"))
    st.checkbox(t(lang, "audio"), key="p_preserve_audio")
    if output_path is not None and output_path.is_file():
        st.download_button(
            t(lang, "download"),
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="video/mp4",
            use_container_width=True,
        )


def _render_timeline(lang: str) -> None:
    st.markdown(
        f"""
        <div class="timeline">
          <div class="timeline-label">{t(lang, "timeline")}</div>
          <div class="timeline-track"><div class="timeline-block"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Motion Depth Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_state()
    lang = st.session_state.lang
    _inject_styles(st.session_state.theme)
    store = _store()
    jobs = store.list_jobs()
    current_job = store.get(st.session_state.current_job_id) if st.session_state.current_job_id else None

    _render_header(lang)

    show_params = bool(st.session_state.show_params)
    if show_params:
        left, center, drawer = st.columns([1.15, 3.35, 1.5], gap="medium")
    else:
        left, center = st.columns([1.15, 4.85], gap="medium")
        drawer = None

    with left:
        _render_history(lang, store, jobs)

    source_path: Path | None = None
    output_path: Path | None = None
    if current_job is not None:
        if current_job.source_exists():
            source_path = Path(current_job.source_path)
        if current_job.output_exists():
            output_path = Path(current_job.output_path)
    elif st.session_state.draft_source and Path(st.session_state.draft_source).is_file():
        source_path = Path(st.session_state.draft_source)

    info = None
    if source_path is not None:
        try:
            info = inspect_video(source_path)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    with center:
        title_l, title_r = st.columns([3.4, 1.1])
        with title_l:
            st.markdown(
                f'<div class="panel-label">{t(lang, "source")} / {t(lang, "depth")}</div>',
                unsafe_allow_html=True,
            )
        with title_r:
            if st.button(t(lang, "params"), key="toggle_params", use_container_width=True):
                st.session_state.show_params = not st.session_state.show_params
                st.rerun()

        src_col, action_col, depth_col = st.columns([1.35, 0.55, 1.35], gap="small")
        with src_col:
            st.markdown(f'<div class="panel-label">{t(lang, "source")}</div>', unsafe_allow_html=True)
            if current_job is None:
                uploaded = st.file_uploader(
                    t(lang, "upload"),
                    type=["mp4", "mov", "avi", "mkv", "webm"],
                    label_visibility="collapsed",
                    key="draft_uploader",
                )
                if uploaded is not None and st.session_state.draft_upload_name != uploaded.name:
                    st.session_state.draft_source = str(_save_upload(uploaded))
                    st.session_state.draft_upload_name = uploaded.name
                    st.rerun()
            if source_path is not None:
                st.markdown('<div class="monitor-wrap">', unsafe_allow_html=True)
                st.video(str(source_path))
                if info is not None:
                    st.markdown(
                        f'<div class="meta-overlay">{info.width}×{info.height} · '
                        f"{info.fps:.2f}fps · {info.format_duration()} · "
                        f"{info.frame_count or '—'}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            elif current_job is not None:
                st.markdown(
                    f'<div class="monitor-empty"><strong>SOURCE</strong>'
                    f"<p>{t(lang, 'missing_source')}</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="monitor-empty"><strong>SOURCE</strong>'
                    f"<p>{t(lang, 'upload_hint')}</p></div>",
                    unsafe_allow_html=True,
                )

        preview_slot = None
        with action_col:
            st.write("")
            st.write("")
            can_run = source_path is not None
            start = st.button(
                t(lang, "process"),
                type="primary",
                use_container_width=True,
                disabled=not can_run,
            )
            if not can_run:
                st.caption(t(lang, "need_source"))

        with depth_col:
            st.markdown(f'<div class="panel-label">{t(lang, "depth")}</div>', unsafe_allow_html=True)
            preview_slot = st.empty()
            if output_path is not None and not start:
                preview_slot.video(str(output_path))
            elif current_job is not None and not current_job.output_exists():
                preview_slot.markdown(
                    f'<div class="monitor-empty"><strong>{t(lang, "standby")}</strong>'
                    f"<p>{t(lang, 'missing_output')}</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                preview_slot.markdown(
                    f'<div class="monitor-empty"><strong>{t(lang, "standby")}</strong>'
                    f"<p>{t(lang, 'standby_hint')}</p></div>",
                    unsafe_allow_html=True,
                )

        progress_bar = st.empty()
        status = st.empty()
        _render_timeline(lang)

        if start and source_path is not None:
            progress_bar.progress(0)
            result_path = unique_path(OUTPUT_DIR, f"{source_path.stem}_depth")
            preview_dir = OUTPUT_DIR / "previews" / result_path.stem
            params = _current_params()
            config = DepthCaptureConfig(
                model_id=default_model_id(),
                scale=params["scale"],
                stride=params["stride"],
                gamma=params["gamma"],
                crush_percentile=params["crush"],
                temporal_smooth=params["smooth"],
                preserve_audio=params["preserve_audio"],
            )

            def on_progress(update: ProgressInfo) -> None:
                progress_bar.progress(min(max(update.progress, 0.0), 1.0))
                total = update.total_frames or "?"
                status.markdown(
                    f"`FRAME {update.frame_index}/{total}`  "
                    f"`{update.progress * 100:5.1f}%`  "
                    f"`{update.fps:.1f} fps`  "
                    f"`ETA {_format_eta(update.eta_seconds)}`"
                )
                if update.preview_path:
                    preview_slot.image(update.preview_path)

            try:
                with st.spinner(t(lang, "loading")):
                    engine = _load_engine(config.model_id)
                pipeline = DepthPipeline(config, engine=engine)
                result = pipeline.process(
                    input_path=source_path,
                    output_path=result_path,
                    progress_callback=on_progress,
                    preview_dir=preview_dir,
                )
                job = store.add(
                    title=source_path.stem,
                    source_path=source_path,
                    output_path=result,
                    params=params,
                    preview_dir=preview_dir,
                )
                st.session_state.current_job_id = job.id
                st.session_state.draft_source = None
                st.session_state.draft_upload_name = None
                progress_bar.progress(1.0)
                status.success(t(lang, "done"))
                preview_slot.video(str(result))
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    if drawer is not None:
        with drawer:
            _render_params(lang, output_path)


if __name__ == "__main__":
    main()
