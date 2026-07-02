#!/usr/bin/env python3
"""
MemorySafe — live governance demo wired to the governed replay buffer.

  streamlit run demo_streamlit.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from demo_engine import get_or_create_session

ROOT = Path(__file__).parent
LOGO = ROOT.parent / "memorysafe-sandbox" / "public" / "memorysafe-logo.png"

BRAND = {
    "bg": "#020408",
    "surface": "#0A0E17",
    "cyan": "#00F0FF",
    "blue": "#2F7BFF",
    "danger": "#FF2A6D",
    "success": "#4ade80",
    "text": "#FFFFFF",
    "body": "#94A3B8",
    "dim": "#475569",
    "border": "rgba(255, 255, 255, 0.08)",
    "glass": "rgba(255, 255, 255, 0.02)",
}

STATUS = {
    "high": (BRAND["danger"], "HIGH RISK"),
    "medium": (BRAND["blue"], "MEDIUM"),
    "stable": (BRAND["success"], "STABLE"),
}

ACTION = {
    "protect": (BRAND["cyan"], "PROTECT"),
    "replay": (BRAND["blue"], "REPLAY"),
    "ignore": (BRAND["dim"], "IGNORE"),
}


def inject_theme() -> None:
    c = BRAND
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        .stApp {{
            background: {c['bg']};
            font-family: Inter, ui-sans-serif, system-ui, sans-serif;
            color: {c['text']};
        }}
        header[data-testid="stHeader"], #MainMenu, footer, .stDeployButton {{ visibility: hidden; }}
        .block-container {{ padding-top: 1.5rem; max-width: 1280px; }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:5px 10px;border-radius:999px;'
        f'font-family:JetBrains Mono,monospace;font-size:0.72rem;letter-spacing:0.06em;'
        f'color:{color};background:{color}1f;border:1px solid {color}33;">{label}</span>'
    )


def metric_card(label: str, value: str, accent: str, glow: str = "none") -> str:
    return f"""
    <div style="background:{BRAND['glass']};border:1px solid {BRAND['border']};
        border-radius:16px;padding:16px 18px;box-shadow:{glow};">
        <div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;
            letter-spacing:0.08em;text-transform:uppercase;color:{BRAND['body']};">{label}</div>
        <div style="font-family:JetBrains Mono,monospace;font-size:2rem;font-weight:800;
            color:{accent};margin-top:6px;line-height:1;">{value}</div>
    </div>
    """


def sample_card(sample: dict) -> str:
    st_color, st_label = STATUS.get(sample["status"], (BRAND["body"], sample["status"].upper()))
    ac_color, ac_label = ACTION.get(sample["action"], (BRAND["body"], sample["action"].upper()))
    mvi_color = BRAND["danger"] if sample["status"] == "high" else BRAND["text"]
    glow = "0 0 20px rgba(255, 42, 109, 0.16)" if sample["status"] == "high" else "0 10px 30px rgba(0,0,0,0.35)"
    return f"""
    <div style="background:{BRAND['surface']};border:1px solid {BRAND['border']};
        border-radius:16px;padding:18px 20px;margin-bottom:1.25rem;box-shadow:{glow};">
        <div style="display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;">
            <div>
                <div style="font-size:1.2rem;font-weight:700;margin-bottom:10px;">{sample['sample_id']}</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    {badge(st_label, st_color)}
                    {badge(sample['rarity'].upper(), BRAND['body'])}
                    {badge(ac_label, ac_color)}
                </div>
            </div>
            <div style="text-align:right;min-width:90px;">
                <div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;
                    letter-spacing:0.08em;color:{BRAND['body']};">MVI</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:2.2rem;
                    font-weight:800;color:{mvi_color};">{sample['mvi']:.2f}</div>
            </div>
        </div>
    </div>
    """


def why_text(sample: dict) -> str:
    action = sample.get("action", "ignore")
    if action == "protect":
        if sample.get("rarity") == "rare":
            return "Rare class + high MVI — quota slot protected in bounded buffer."
        return "Old-task memory with high ProtectScore — shell retention active."
    if action == "replay":
        return f"Fragility signal — weighted replay (seen {sample.get('seen', 0)}×)."
    return "Low governance spend — monitored, not prioritized for replay."


@st.fragment(run_every=1.5)
def live_dashboard() -> None:
    session = get_or_create_session(st.session_state, seed=42)
    snap = session.tick()

    visible = snap["samples"]
    if not visible:
        st.info("Buffer warming up — governance kicks in after first batches.")
        return

    selected = max(visible, key=lambda s: s.get("protect", s.get("mvi", 0)))

    protected = snap["protected"]
    replay_n = snap["replay_n"]
    high_n = snap["high_n"]
    pos_pct = round(100 * snap["pos_in_buffer"] / max(snap["buffer_fill"], 1))
    fri_pct = round(snap["fri"] * 100)

    # Header bar
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
            border:1px solid {BRAND['border']};border-radius:16px;background:rgba(0,0,0,0.3);
            padding:14px 20px;margin-bottom:16px;backdrop-filter:blur(12px);">
            <div style="display:flex;align-items:center;gap:14px;">
                <span style="font-size:1.15rem;font-weight:800;">MemorySafe</span>
                <span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                    letter-spacing:0.14em;padding:4px 10px;border-radius:999px;
                    color:{BRAND['cyan']};border:1px solid rgba(0,240,255,0.2);
                    background:rgba(0,240,255,0.1);">LIVE BUFFER</span>
                <span style="color:{BRAND['body']};font-size:0.9rem;">Predictive Memory Systems</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="display:inline-flex;align-items:center;gap:6px;font-family:JetBrains Mono,monospace;
                    font-size:0.68rem;letter-spacing:0.12em;color:{BRAND['cyan']};
                    border:1px solid rgba(0,240,255,0.2);background:rgba(0,240,255,0.08);
                    padding:6px 12px;border-radius:999px;">
                    <span style="width:7px;height:7px;border-radius:50%;background:{BRAND['cyan']};
                        box-shadow:0 0 10px rgba(0,240,255,0.8);"></span>
                    MONITORING ACTIVE
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    center, right = st.columns([0.62, 0.38], gap="medium")

    with center:
        st.markdown(
            f"""
            <div style="margin-bottom:14px;">
                <div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                    letter-spacing:0.12em;color:{BRAND['body']};">MEMORY OBSERVABILITY LAYER</div>
                <div style="font-size:1.5rem;font-weight:700;margin-top:4px;">System Overview</div>
                <div style="color:{BRAND['body']};font-size:0.9rem;margin-top:4px;">
                    Only the signals that matter right now.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("High Risk", str(high_n), BRAND["danger"], "0 0 20px rgba(255,42,109,0.12)"), unsafe_allow_html=True)
        m2.markdown(metric_card("Protected", str(protected), BRAND["cyan"], "0 0 20px rgba(0,240,255,0.14)"), unsafe_allow_html=True)
        m3.markdown(metric_card("Replay", str(replay_n), BRAND["blue"]), unsafe_allow_html=True)
        m4.markdown(metric_card("FRI", f"{fri_pct}%", BRAND["success"]), unsafe_allow_html=True)

        auprc_line = ""
        if snap.get("combined_auprc") is not None:
            auprc_line = (
                f"Combined AUPRC <span style='color:{BRAND['cyan']};font-weight:700;'>"
                f"{snap['combined_auprc']:.3f}</span> &nbsp;·&nbsp; "
            )

        st.markdown(
            f"""
            <div style="background:{BRAND['glass']};border:1px solid {BRAND['border']};
                border-radius:16px;padding:14px 16px;margin:12px 0;">
                <div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                    letter-spacing:0.12em;color:{BRAND['body']};margin-bottom:6px;">GOVERNED BUFFER</div>
                <div style="color:{BRAND['body']};font-size:0.88rem;">
                    Wave <span style="color:{BRAND['cyan']};font-weight:700;">{snap['wave']}/{snap['total_waves']}</span>
                    &nbsp;·&nbsp; Fill <span style="color:{BRAND['cyan']};font-weight:700;">{snap['buffer_fill']}/{snap['buffer_cap']}</span>
                    &nbsp;·&nbsp; Pos quota <span style="color:{BRAND['cyan']};font-weight:700;">{pos_pct}%</span>
                    <br/>{auprc_line}MVI μ <span style="color:{BRAND['cyan']};font-weight:700;">{snap['mean_mvi']:.2f}</span>
                    &nbsp;·&nbsp; {len(visible)} top slots shown
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for sample in visible:
            st.markdown(sample_card(sample), unsafe_allow_html=True)

    with right:
        ac_color, ac_label = ACTION.get(selected["action"], (BRAND["body"], selected["action"]))
        st_color, st_label = STATUS.get(selected["status"], (BRAND["body"], selected["status"]))
        mvi_color = BRAND["danger"] if selected["status"] == "high" else BRAND["text"]

        st.markdown(
            f"""
            <div style="background:{BRAND['glass']};border:1px solid {BRAND['border']};
                border-radius:16px;padding:16px;">
                <div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                    letter-spacing:0.12em;color:{BRAND['body']};">INSPECTOR</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:1.2rem;
                    font-weight:700;margin-top:8px;">{selected['sample_id']}</div>
                <div style="color:{BRAND['body']};font-size:0.85rem;margin:4px 0 14px 0;">Focused sample</div>
                <div style="margin-bottom:16px;">{badge(st_label, st_color)} {badge(ac_label, ac_color)}</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                    letter-spacing:0.08em;color:{BRAND['body']};">MVI</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:2.6rem;font-weight:800;
                    color:{mvi_color};line-height:1;margin-top:4px;">{selected['mvi']:.2f}</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                    letter-spacing:0.08em;color:{BRAND['body']};margin-top:12px;">PROTECT</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:1.4rem;font-weight:700;
                    color:{BRAND['cyan']};">{selected.get('protect', 0):.2f}</div>
                <p style="color:{BRAND['body']};font-size:0.9rem;line-height:1.55;margin:16px 0 0 0;">
                    {why_text(selected)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background:{BRAND['surface']};border:1px solid {BRAND['border']};
                border-radius:16px;padding:16px;margin-top:14px;">
                <div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                    letter-spacing:0.12em;color:{BRAND['body']};">SYSTEM STATUS</div>
                <p style="color:{BRAND['text']};margin:10px 0 4px 0;">Buffer <span style="color:{BRAND['cyan']};">Governed replay</span></p>
                <p style="color:{BRAND['text']};margin:4px 0 0 0;">Replay step <span style="color:{BRAND['success'] if snap['last_replayed'] else BRAND['body']};">{'Yes' if snap['last_replayed'] else 'No'}</span></p>
                <p style="color:{BRAND['text']};margin:4px 0 0 0;">Source <span style="color:{BRAND['success']};">Governed CL sim</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="MemorySafe", page_icon="🧠", layout="wide")
    inject_theme()

    st.markdown(
        f"""
        <div style="margin-bottom:18px;">
            <div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;
                letter-spacing:0.1em;text-transform:uppercase;color:{BRAND['body']};">
                Real-time memory governance
            </div>
            <h1 style="margin:6px 0 0 0;font-size:clamp(2rem,4vw,3.2rem);font-weight:800;
                letter-spacing:-0.04em;line-height:1.05;">
                <span style="background:linear-gradient(90deg,{BRAND['text']},{BRAND['cyan']});
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">MemorySafe</span>
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_dashboard()

    if LOGO.exists():
        st.markdown("<div style='margin-top:36px;text-align:center;'>", unsafe_allow_html=True)
        st.image(str(LOGO), width=160)
        st.markdown(
            f"<p style='color:{BRAND['dim']};font-family:JetBrains Mono,monospace;"
            f"font-size:0.78rem;margin-top:8px;'>predictive memory systems · live governed buffer</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='text-align:center;color:{BRAND['cyan']};margin-top:2rem;'>memorysafe.ca</p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
