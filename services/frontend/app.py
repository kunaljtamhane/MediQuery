"""
Person D — Streamlit Frontend (Weeks 7-8)
Provides a research assistant UI with:
  - Search bar with streaming LLM response
  - Source attribution panel
  - Multi-turn conversation history
  - Deduplication indicator (new vs. previously-seen sources)
"""
import streamlit as st
import httpx
import json
import os

SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://spring_boot_api:8080")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Session state initialisation ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # list of {role, content, sources}
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "seen_sources" not in st.session_state:
    st.session_state.seen_sources = set()  # track deduplication


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("AI Research Assistant")
st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

col_chat, col_sources = st.columns([2, 1])

with col_chat:
    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_query := st.chat_input("Ask a research question..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})

        # Stream assistant response
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_tokens = []
            sources_from_stream = []

            try:
                with httpx.stream(
                    "POST",
                    f"{SPRING_BOOT_URL}/search",
                    json={"query": user_query, "sessionId": st.session_state.session_id},
                    timeout=60.0,
                ) as response:
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line.removeprefix("data:").strip()
                        if payload == "[DONE]":
                            break
                        data = json.loads(payload)
                        if "sources" in data:
                            sources_from_stream = data["sources"]
                        if "token" in data:
                            answer_tokens.append(data["token"])
                            answer_placeholder.markdown("".join(answer_tokens) + "▌")

            except Exception as e:
                answer_tokens = [f"Connection error: {e}. Make sure services are running."]

            final_answer = "".join(answer_tokens)
            answer_placeholder.markdown(final_answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "sources": sources_from_stream,
        })
        st.rerun()

with col_sources:
    st.subheader("Sources")

    # Show sources from the latest assistant message
    last_sources = []
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and msg.get("sources"):
            last_sources = msg["sources"]
            break

    if last_sources:
        for src in last_sources:
            doc_id = src.get("doc_id", "unknown")
            title = src.get("title") or doc_id
            score = src.get("score", 0)

            is_new = doc_id not in st.session_state.seen_sources
            st.session_state.seen_sources.add(doc_id)

            badge = "NEW" if is_new else "seen"
            color = "green" if is_new else "gray"

            st.markdown(
                f":{color}[**{badge}**] **{title[:60]}**  \n"
                f"Score: `{score:.3f}` | ID: `{doc_id[:20]}...`"
            )
            st.divider()
    else:
        st.info("Sources will appear here after your first query.")

    # Conversation stats
    st.subheader("Session Stats")
    n_turns = len([m for m in st.session_state.messages if m["role"] == "user"])
    n_unique_sources = len(st.session_state.seen_sources)
    st.metric("Turns", n_turns)
    st.metric("Unique sources seen", n_unique_sources)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.seen_sources = set()
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
