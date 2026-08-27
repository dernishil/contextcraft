import os
import requests
import streamlit as st

st.set_page_config(page_title="ContextCraft", layout="wide")

API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.markdown("### ContextCraft")
st.caption("Prompt resolver and contextual memory pipeline")

if "logs" not in st.session_state:
  st.session_state.logs = []
if "history" not in st.session_state:
  st.session_state.history = []

with st.sidebar:
  st.markdown("#### Context Memory")
  doc_input = st.text_area(
      "Store fact or background",
      placeholder="e.g. Current stack is FastAPI + PyTorch on Arch Linux",
      height=100,
  )
  if st.button("Save Entry", use_container_width=True):
    if doc_input.strip():
      res = requests.post(
          f"{API_BASE}/api/v1/memory", json={"content": doc_input}
      )
      if res.status_code == 200:
        st.toast("Saved context")
      else:
        st.error("Error writing to store")

  if st.button("Purge Database", type="secondary", use_container_width=True):
    requests.delete(f"{API_BASE}/api/v1/memory")
    st.toast("Database purged")

  st.divider()
  st.markdown("#### Existing Docs")
  try:
    data = requests.get(f"{API_BASE}/api/v1/memory", timeout=2).json()
    for item in data.get("data", []):
      st.text(f"• {item['text']}")
  except Exception:
    st.caption("Server offline.")

col_left, col_right = st.columns(2)

with col_left:
  user_query = st.text_input("Input query", placeholder="e.g. fix that error")
  submit = st.button("Send", type="primary")

if submit and user_query.strip():
  payload = {"text": user_query, "history": st.session_state.history}
  try:
    res = requests.post(f"{API_BASE}/api/v1/chat", json=payload).json()
    st.session_state.history.append(f"User: {user_query}")
    st.session_state.logs.append(res)
  except Exception as err:
    st.error(f"Request failed: {err}")

with col_right:
  if st.session_state.logs:
    latest = st.session_state.logs[-1]
    st.markdown("#### Execution Pipeline")
    st.text_area("Resolved Prompt", latest.get("resolved"), height=100)
    if latest.get("context"):
      st.markdown("**Matched Context:**")
      for c in latest["context"]:
        st.caption(f"- {c}")

st.divider()
st.markdown("#### Thread")
for item in reversed(st.session_state.logs):
  with st.chat_message("user"):
    st.write(item["raw"])
  with st.chat_message("assistant"):
    st.write(item["response"])