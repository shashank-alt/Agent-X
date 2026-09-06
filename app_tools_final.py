from chatbot_tools_backend_final import (
    chatbot,
    get_all_threads,
    ingest_rag_document,
    save_thread_title,
    get_all_thread_titles,
    llm
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖"
)

st.title("Agent X")


# ============================================================
# THREAD MANAGEMENT
# ============================================================

# Generate a new thread id for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

# ============================================================
# THREAD TITLE GENERATION
# ============================================================

def generate_thread_title(user_input):
    """
    Generate a short title for a conversation
    based on the user's first message.
    """

    # Use the existing LLM from backend

    try:

        response = llm.invoke(
            f"""
Generate a short title for a chatbot conversation.

User's message:
{user_input}

Rules:
- Maximum 5 words
- Keep it concise
- Do not use quotation marks
- Do not add explanations
- Return only the title
"""
        )

        title = response.content.strip()

        # Remove accidental quotation marks
        title = title.replace('"', "").replace("'", "")

        # Safety fallback
        if not title:
            return "New Conversation"

        return title[:60]

    except Exception:

        # If title generation fails
        # use the beginning of user's message
        title = user_input.strip()

        if len(title) > 35:
            title = title[:35] + "..."

        return title or "New Conversation"


# Add a new thread id to the conversation list
def add_thread(thread_id):

    # Prevent same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new conversation
def reset_chat():

    # Generate and assign new thread_id
    st.session_state["thread_id"] = generate_thread_id()

    # Clear current message history from UI
    st.session_state["message_history"] = []

    # ========================= HITL ADDED =========================

    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None

    # =============================================================

    # Add new thread to the list of threads
    add_thread(st.session_state["thread_id"])


# Load previous conversation history from LangGraph checkpointer
def load_conversation(thread_id):

    # Get saved state for selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    return state.values.get("messages", [])


# ============================================================
# HITL HELPER FUNCTIONS
# ============================================================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # Newly created thread may not have checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):

    """
    Save pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):

    """
    Synchronize Streamlit HITL state with LangGraph checkpoint.

    This allows pending approval to reappear after:
    - Streamlit rerun
    - browser refresh
    - switching conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):

            st.session_state["pending_hitl"] = None


# ============================================================
# HITL RESUME FUNCTION
# ============================================================

def resume_hitl_execution(decision):

    """
    Resume an interrupted LangGraph execution.

    decision:
        "yes" -> approve
        "no"  -> reject
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Thread that originally triggered interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # IMPORTANT:
    # Resume using the SAME thread ID
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace"
    }

    try:

        with st.chat_message("assistant"):

            status_box = st.status(
                "🔄 Resuming requested action...",
                expanded=True
            )

            def resumed_ai_stream():

                # Resume interrupted graph
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages"
                ):

                    # -----------------------------------------
                    # Tool result
                    # -----------------------------------------

                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_box.write(
                            f"🔧 `{tool_name}` completed"
                        )

                    # -----------------------------------------
                    # Stream assistant response
                    # -----------------------------------------

                    if isinstance(
                        message_chunk,
                        (AIMessage, AIMessageChunk)
                    ):

                        content = message_chunk.text

                        if isinstance(content, str):

                            if content:
                                yield content

                        elif isinstance(content, list):

                            for item in content:

                                if isinstance(item, dict):

                                    if item.get("type") == "text":

                                        text = item.get("text")

                                        if text:
                                            yield text

            # Display resumed assistant response
            resumed_ai_message = st.write_stream(
                resumed_ai_stream()
            )

            # Check if another interrupt occurred
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_box.update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No pending HITL
                st.session_state["pending_hitl"] = None

                status_box.update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        # Save resumed assistant response
        if resumed_ai_message:

            st.session_state["message_history"].append(
                {
                    "role": "assistant",
                    "content": resumed_ai_message
                }
            )

        # Rerun so UI becomes consistent
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )


# ============================================================
# SESSION STATE
# ============================================================

# Initialize message history
if "message_history" not in st.session_state:

    st.session_state["message_history"] = []

# Initialize chat threads
if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = get_all_threads()

#  Initializing thread titles
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = get_all_thread_titles()


# Initialize thread id
if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()

    add_thread(
        st.session_state["thread_id"]
    )


# ========================= HITL =========================

# Initialize pending HITL
if "pending_hitl" not in st.session_state:

    st.session_state["pending_hitl"] = None


# ============================================================
# SYNC CURRENT THREAD
# ============================================================

add_thread(
    st.session_state["thread_id"]
)


# Recover pending approval after refresh/rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)


# ============================================================
# CONFIG
# ============================================================

config = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    }
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Chat History")


# New conversation button
if st.sidebar.button("+"):

    reset_chat()

    st.rerun()


# Display conversations
for thread_id in st.session_state["chat_threads"][::-1]:
    thread_title = st.session_state["thread_titles"].get(
            thread_id,
            "New Conversation"
        )

    if st.sidebar.button(
        thread_title,
        key=thread_id
    ):
        # Select thread
        st.session_state["thread_id"] = thread_id

        # Load saved conversation
        messages = load_conversation(
            thread_id
        )

        temp_messages = []

        for message in messages:

            # User message
            if isinstance(
                message,
                HumanMessage
            ):

                role = "user"

            # Assistant message
            elif isinstance(
                message,
                AIMessage
            ):

                role = "assistant"

            # Ignore tool/system messages
            else:

                continue

            temp_messages.append(
                {
                    "role": role,
                    "content": message.text
                }
            )

        # Replace UI history
        st.session_state["message_history"] = temp_messages

        # Restore pending HITL
        sync_pending_interrupt(
            thread_id
        )

        st.rerun()


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state["message_history"]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# HITL APPROVAL INTERFACE
# ============================================================

pending_hitl = st.session_state.get(
    "pending_hitl"
)


# Check whether pending approval belongs to current thread
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and
    pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


# Display approval controls
if current_thread_has_pending_hitl:

    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    # Approve
    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):

            resume_hitl_execution(
                "yes"
            )

    # Reject
    with reject_column:

        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):

            resume_hitl_execution(
                "no"
            )


# ============================================================
# CHAT INPUT + PDF UPLOAD
# ============================================================

submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl
)


user_input = None


# ============================================================
# PROCESS USER INPUT / PDF
# ============================================================

if submission:

    # Get text
    user_input = submission.text

    # Get uploaded files
    uploaded_files = submission.files


    # --------------------------------------------------------
    # PDF RAG INGESTION
    # --------------------------------------------------------

    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        temporary_file_path = None

        try:

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_pdf.getvalue()
                )

                temporary_file_path = (
                    temporary_file.name
                )


            # Process PDF
            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )


            st.toast(
                f"{uploaded_pdf.name} processed successfully.",
                icon="✅"
            )


        except Exception as error:

            st.error(
                f"PDF processing failed: {error}"
            )


        finally:

            # Delete temporary PDF
            if (
                temporary_file_path
                and
                os.path.exists(
                    temporary_file_path
                )
            ):

                os.remove(
                    temporary_file_path
                )


# ============================================================
# CHAT EXECUTION
# ============================================================

if user_input:

    current_thread_id = st.session_state["thread_id"]

    # ========================================================
    # GENERATE TITLE FOR NEW CONVERSATION
    # ========================================================

    if current_thread_id not in st.session_state["thread_titles"]:

        st.session_state["thread_titles"][
            current_thread_id
        ] = generate_thread_title(user_input)


    # ========================================================
    # STORE USER MESSAGE
    # ========================================================

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # Display user message
    with st.chat_message("user"):

        st.write(
            user_input
        )


    # --------------------------------------------------------
    # LANGGRAPH CONFIG
    # --------------------------------------------------------

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace"
    }


    # --------------------------------------------------------
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }


        def ai_only_stream():

            used_tools = set()


            # Stream LangGraph
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):


                # ==================================================
                # TOOL CALL DETECTION
                # ==================================================

                if hasattr(
                    message_chunk,
                    "tool_call_chunks"
                ):

                    for tool_call in (
                        message_chunk.tool_call_chunks
                    ):

                        tool_name = tool_call.get(
                            "name"
                        )

                        if (
                            tool_name
                            and
                            tool_name not in used_tools
                        ):

                            used_tools.add(
                                tool_name
                            )

                            # Create status box
                            if (
                                status_holder["box"]
                                is None
                            ):

                                status_holder["box"] = st.status(
                                    f"🔧 Using `{tool_name}`...",
                                    expanded=True
                                )

                            else:

                                status_holder["box"].write(
                                    f"🔧 Using `{tool_name}`..."
                                )


                # ==================================================
                # TOOL RESULT
                # ==================================================

                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if (
                        status_holder["box"]
                        is None
                    ):

                        status_holder["box"] = st.status(
                            f"🔧 `{tool_name}` completed",
                            expanded=True
                        )

                    else:

                        status_holder["box"].write(
                            f"✅ `{tool_name}` completed"
                        )


                # ==================================================
                # STREAM ASSISTANT TEXT
                # ==================================================

                if isinstance(
                    message_chunk,
                    (AIMessage, AIMessageChunk)
                ):

                    content = (
                        message_chunk.text
                    )


                    # Gemini simple text response
                    if isinstance(
                        content,
                        str
                    ):

                        if content:

                            yield content


                    # Gemini structured content
                    elif isinstance(
                        content,
                        list
                    ):

                        for item in content:

                            if isinstance(
                                item,
                                dict
                            ):

                                if (
                                    item.get("type")
                                    == "text"
                                ):

                                    text = item.get(
                                        "text"
                                    )

                                    if text:

                                        yield text


            # ==================================================
            # HITL DETECTION
            # ==================================================

            pending_interrupt = (
                get_pending_interrupt(
                    st.session_state["thread_id"]
                )
            )


            if pending_interrupt is not None:

                # Save pending approval
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )


                # Show message to user
                yield (
                    "\n\n⚠️ This stock purchase "
                    "requires your approval. "
                    "Use the Approve Purchase "
                    "or Reject Purchase button below."
                )


        # ==================================================
        # DISPLAY STREAM
        # ==================================================

        ai_message = st.write_stream(
            ai_only_stream()
        )


        # ==================================================
        # UPDATE TOOL STATUS
        # ==================================================

        if status_holder["box"] is not None:

            # Check whether graph is waiting for approval
            if get_pending_interrupt(
                st.session_state["thread_id"]
            ) is not None:

                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )

            else:

                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )


    # ========================================================
    # RERUN FOR HITL
    # ========================================================

    if (
        st.session_state.get(
            "pending_hitl"
        ) is not None
        and
        st.session_state["pending_hitl"].get(
            "thread_id"
        )
        == st.session_state["thread_id"]
    ):

        st.rerun()