import json

import requests
import streamlit as st

DEFAULT_API_BASE_URL = "http://localhost"

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")


def init_session_state():
    defaults = {
        "api_base_url": DEFAULT_API_BASE_URL,
        "token": None,
        "email": None,
        "conversation_id": None,
        "messages": [],  # [{role, content, matches}]
        "auth_mode": "Login",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_url(path: str) -> str:
    return f"{st.session_state.api_base_url.rstrip('/')}{path}"


def logout():
    st.session_state.token = None
    st.session_state.email = None
    st.session_state.conversation_id = None
    st.session_state.messages = []


def login(email: str, password: str) -> str | None:
    """Log in and store the JWT. Returns an error message on failure, else None."""
    try:
        resp = requests.post(
            api_url("/api/auth/jwt/login"),
            data={"username": email, "password": password},
            timeout=30,
        )
    except requests.RequestException as e:
        return f"Could not reach API: {e}"
    if resp.status_code != 200:
        try:
            return resp.json().get("detail", resp.text)
        except ValueError:
            return resp.text
    st.session_state.token = resp.json()["access_token"]
    st.session_state.email = email
    return None


def register(email: str, password: str) -> str | None:
    """Register a new user. Returns an error message on failure, else None."""
    try:
        resp = requests.post(
            api_url("/api/auth/register"),
            json={"email": email, "password": password},
            timeout=30,
        )
    except requests.RequestException as e:
        return f"Could not reach API: {e}"
    if resp.status_code not in (200, 201):
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return str(detail)
    return None


def fetch_files() -> list[dict]:
    resp = requests.get(api_url("/api/files/list"), headers=auth_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()["files"]


def upload_file(uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    resp = requests.post(
        api_url("/api/files/hybrid_upload"), headers=auth_headers(), files=files, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def delete_file(file_id: str):
    resp = requests.delete(api_url(f"/api/files/{file_id}"), headers=auth_headers(), timeout=30)
    resp.raise_for_status()


def fetch_conversations() -> list[dict]:
    resp = requests.get(api_url("/api/conversations"), headers=auth_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()["conversations"]


def create_conversation() -> str:
    resp = requests.post(api_url("/api/conversations"), headers=auth_headers(), json={}, timeout=30)
    resp.raise_for_status()
    return resp.json()["conversation_id"]


def load_conversation(conversation_id: str) -> list[dict]:
    resp = requests.get(
        api_url(f"/api/conversations/{conversation_id}"), headers=auth_headers(), timeout=30
    )
    resp.raise_for_status()
    turns = resp.json()["turns"]
    messages = []
    for turn in turns:
        messages.append({"role": turn["role"], "content": turn["content"], "matches": []})
    return messages


def stream_answer(query: str, conversation_id: str | None, placeholder):
    """Stream an SSE response, live-updating `placeholder`. Returns (answer_text, matches)."""
    params = {"query": query}
    if conversation_id:
        params["conversation_id"] = conversation_id

    answer = ""
    matches: list[dict] = []
    event = None

    with requests.post(
        api_url("/api/search/stream"),
        headers=auth_headers(),
        params=params,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None or raw_line == "":
                continue
            if raw_line.startswith("event:"):
                event = raw_line[len("event:"):].strip()
            elif raw_line.startswith("data:"):
                data_str = raw_line[len("data:"):].strip()
                data = json.loads(data_str)
                if event == "matches":
                    matches = data
                elif event == "token":
                    # live stream sends {"token": "..."}; cached hits send the full answer string
                    token = data["token"] if isinstance(data, dict) else data
                    answer += token
                    placeholder.markdown(answer)
                elif event == "error":
                    st.error(data.get("detail", "Unknown error"))
                elif event == "done":
                    break
    return answer, matches


def render_sidebar_auth():
    st.sidebar.title("Document Assistant")
    st.session_state.auth_mode = st.sidebar.radio(
        "Account", ["Login", "Register"], horizontal=True,
        index=["Login", "Register"].index(st.session_state.auth_mode),
    )
    with st.sidebar.form("auth_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(st.session_state.auth_mode)
        if submitted:
            if st.session_state.auth_mode == "Login":
                error = login(email, password)
                if error:
                    st.sidebar.error(error)
                else:
                    st.rerun()
            else:
                error = register(email, password)
                if error:
                    st.sidebar.error(error)
                else:
                    st.sidebar.success("Registered. Check your email to verify your account, then log in.")

    with st.sidebar.expander("API settings"):
        st.session_state.api_base_url = st.text_input("API base URL", value=st.session_state.api_base_url)


def render_sidebar_files():
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.email}")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Files")

    uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    if uploaded_file is not None and st.sidebar.button("Upload"):
        try:
            result = upload_file(uploaded_file)
            st.sidebar.success(f"{result['file_name']}: {result['status']}")
        except requests.HTTPError as e:
            st.sidebar.error(f"Upload failed: {e.response.json().get('detail', e.response.text)}")

    try:
        files = fetch_files()
    except requests.HTTPError as e:
        st.sidebar.error(f"Could not list files: {e}")
        files = []

    for f in files:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(f"{f['file_name']} ({f['status']})")
        if col2.button("🗑", key=f"delete_{f['file_id']}"):
            try:
                delete_file(f["file_id"])
                st.rerun()
            except requests.HTTPError as e:
                st.sidebar.error(f"Delete failed: {e}")


def render_sidebar_conversations():
    st.sidebar.divider()
    st.sidebar.subheader("Conversations")

    if st.sidebar.button("+ New conversation"):
        st.session_state.conversation_id = create_conversation()
        st.session_state.messages = []
        st.rerun()

    try:
        conversations = fetch_conversations()
    except requests.HTTPError as e:
        st.sidebar.error(f"Could not list conversations: {e}")
        conversations = []

    if conversations:
        labels = {c["conversation_id"]: (c["title"] or c["conversation_id"][:8]) for c in conversations}
        ids = list(labels.keys())
        current = st.session_state.conversation_id
        index = ids.index(current) if current in ids else 0
        selected = st.sidebar.selectbox(
            "Select conversation", ids, index=index, format_func=lambda cid: labels[cid]
        )
        if selected != st.session_state.conversation_id:
            st.session_state.conversation_id = selected
            st.session_state.messages = load_conversation(selected)
            st.rerun()


def render_chat():
    st.title("Ask your documents")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("matches"):
                with st.expander("Sources"):
                    for m in message["matches"]:
                        st.markdown(f"**{m['file']}** (page {m.get('page')}, score {m['score']:.3f})")
                        st.caption(m["text"])

    query = st.chat_input("Ask a question about your documents")
    if query:
        st.session_state.messages.append({"role": "user", "content": query, "matches": []})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                answer, matches = stream_answer(query, st.session_state.conversation_id, placeholder)
            except requests.HTTPError as e:
                answer, matches = f"Error: {e}", []
                placeholder.markdown(answer)
            if matches:
                with st.expander("Sources"):
                    for m in matches:
                        st.markdown(f"**{m['file']}** (page {m.get('page')}, score {m['score']:.3f})")
                        st.caption(m["text"])
        st.session_state.messages.append({"role": "assistant", "content": answer, "matches": matches})


def main():
    init_session_state()

    if not st.session_state.token:
        render_sidebar_auth()
        st.title("Document Assistant")
        st.info("Log in or register from the sidebar to get started.")
        return

    render_sidebar_files()
    render_sidebar_conversations()
    render_chat()


if __name__ == "__main__":
    main()
