# import tritonclient.grpc as grpcclient
# import numpy as np
# import json
# import time

# TRITON_URL = "localhost:8001"
# MODEL_NAME = "Llama-3.2-1B-Instruct"

# client = grpcclient.InferenceServerClient(url=TRITON_URL)

# print("🤖 Llama-3.2-1B-Instruct Chatbot (type 'exit' to quit)\n")

# # ---------- STREAM CALLBACK ----------
# def callback(result, error):
#     if error:
#         print("\n❌ Error:", error)
#         return

#     if result is None:
#         return

#     try:
#         output = result.as_numpy("text_output")
#         if output is not None:
#             token = output[0].decode("utf-8")
#             if token.strip():
#                 print(token, end="", flush=True)
#     except Exception:
#         pass


# client.start_stream(callback=callback)

# try:
#     while True:
#         user_input = input("\n👤 You: ")
#         if user_input.lower() in ("exit", "quit"):
#             break

#         # ✅ INSTRUCT MODEL CHAT TEMPLATE
#         prompt = f"""<|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>
# You are a helpful, polite AI assistant.
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# {user_input}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """

#         text_input = grpcclient.InferInput("text_input", [1], "BYTES")
#         text_input.set_data_from_numpy(
#             np.array([prompt.encode("utf-8")], dtype=object)
#         )

#         stream_input = grpcclient.InferInput("stream", [1], "BOOL")
#         stream_input.set_data_from_numpy(np.array([True], dtype=bool))

#         sampling_params = {
#             "temperature": 0.7,
#             "top_p": 0.9,
#             "max_tokens": 256
#         }

#         sampling_input = grpcclient.InferInput(
#             "sampling_parameters", [1], "BYTES"
#         )
#         sampling_input.set_data_from_numpy(
#             np.array([json.dumps(sampling_params).encode("utf-8")], dtype=object)
#         )

#         exclude_input = grpcclient.InferInput(
#             "exclude_input_in_output", [1], "BOOL"
#         )
#         exclude_input.set_data_from_numpy(np.array([True], dtype=bool))

#         print("🤖 Bot: ", end="", flush=True)

#         client.async_stream_infer(
#             model_name=MODEL_NAME,
#             inputs=[
#                 text_input,
#                 stream_input,
#                 sampling_input,
#                 exclude_input
#             ]
#         )

#         time.sleep(5)
#         print()

# except KeyboardInterrupt:
#     pass
# finally:
#     client.stop_stream()
#     print("\n✅ Chat closed")

##################################################################################################################
# import tritonclient.grpc as grpcclient
# import numpy as np
# import json
# import gradio as gr
# import queue

# # ---------------- CONFIG ----------------
# TRITON_URL = "localhost:8001"
# MODEL_NAME = "Llama-3.2-1B-Instruct"

# client = grpcclient.InferenceServerClient(url=TRITON_URL)

# token_queue = queue.Queue()

# # ---------- STREAM CALLBACK ----------
# def callback(result, error):
#     if error:
#         token_queue.put(f"\n❌ Error: {error}")
#         return

#     if result is None:
#         return

#     try:
#         output = result.as_numpy("text_output")
#         if output is not None:
#             token = output[0].decode("utf-8")
#             if token.strip():
#                 token_queue.put(token)
#     except Exception:
#         pass


# client.start_stream(callback=callback)

# # ---------- CHAT FUNCTION ----------
# def chat(user_input, history):
#     token_queue.queue.clear()

#     prompt = f"""<|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>
# You are a helpful, polite AI assistant.
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# {user_input}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """

#     text_input = grpcclient.InferInput("text_input", [1], "BYTES")
#     text_input.set_data_from_numpy(
#         np.array([prompt.encode("utf-8")], dtype=object)
#     )

#     stream_input = grpcclient.InferInput("stream", [1], "BOOL")
#     stream_input.set_data_from_numpy(np.array([True], dtype=bool))

#     sampling_params = {
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "max_tokens": 256
#     }

#     sampling_input = grpcclient.InferInput(
#         "sampling_parameters", [1], "BYTES"
#     )
#     sampling_input.set_data_from_numpy(
#         np.array([json.dumps(sampling_params).encode("utf-8")], dtype=object)
#     )

#     exclude_input = grpcclient.InferInput(
#         "exclude_input_in_output", [1], "BOOL"
#     )
#     exclude_input.set_data_from_numpy(np.array([True], dtype=bool))

#     # Append messages (Gradio 6 format)
#     history = history + [
#         {"role": "user", "content": user_input},
#         {"role": "assistant", "content": ""}
#     ]

#     client.async_stream_infer(
#         model_name=MODEL_NAME,
#         inputs=[
#             text_input,
#             stream_input,
#             sampling_input,
#             exclude_input
#         ]
#     )

#     while True:
#         try:
#             token = token_queue.get(timeout=0.5)
#             history[-1]["content"] += token
#             yield history
#         except queue.Empty:
#             break


# # ---------- GRADIO UI ----------
# with gr.Blocks(title="Llama-3.2-1B-Instruct Chatbot") as demo:
#     gr.Markdown("## 🤖 Llama-3.2-1B-Instruct Chatbot (Triton + vLLM)")

#     chatbot = gr.Chatbot(height=400)  # ✅ NO type argument

#     user_input = gr.Textbox(
#         placeholder="Type your message here...",
#         show_label=False
#     )

#     user_input.submit(chat, [user_input, chatbot], chatbot)
#     user_input.submit(lambda: "", None, user_input)

# demo.launch()

###############################################################################################################################
# import tritonclient.grpc as grpcclient
# import numpy as np
# import json
# import gradio as gr
# import queue

# # ---------------- CONFIG ----------------
# TRITON_URL = "localhost:8001"
# MODEL_NAME = "Llama-3.2-1B-Instruct"

# client = grpcclient.InferenceServerClient(url=TRITON_URL)
# token_queue = queue.Queue()

# # ---------- STREAM CALLBACK ----------
# def callback(result, error):
#     if error:
#         token_queue.put(f"\n❌ Error: {error}")
#         return

#     if result is None:
#         return

#     try:
#         output = result.as_numpy("text_output")
#         if output is not None:
#             token = output[0].decode("utf-8")
#             if token.strip():
#                 token_queue.put(token)
#     except Exception:
#         pass


# client.start_stream(callback=callback)

# # ---------- CHAT FUNCTION ----------
# def chat(
#     user_input,
#     history,
#     temperature,
#     max_tokens,
#     top_p,
#     top_k,
#     presence_penalty,
#     frequency_penalty
# ):
#     token_queue.queue.clear()

#     prompt = f"""<|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>
# You are a helpful, polite AI assistant.
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# {user_input}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """

#     # -------- Triton Inputs --------
#     text_input = grpcclient.InferInput("text_input", [1], "BYTES")
#     text_input.set_data_from_numpy(np.array([prompt.encode()], dtype=object))

#     stream_input = grpcclient.InferInput("stream", [1], "BOOL")
#     stream_input.set_data_from_numpy(np.array([True], dtype=bool))

#     sampling_params = {
#         "temperature": temperature,
#         "max_tokens": int(max_tokens),
#         "top_p": top_p,
#         "top_k": int(top_k),
#         "presence_penalty": presence_penalty,
#         "frequency_penalty": frequency_penalty
#     }

#     sampling_input = grpcclient.InferInput(
#         "sampling_parameters", [1], "BYTES"
#     )
#     sampling_input.set_data_from_numpy(
#         np.array([json.dumps(sampling_params).encode()], dtype=object)
#     )

#     exclude_input = grpcclient.InferInput(
#         "exclude_input_in_output", [1], "BOOL"
#     )
#     exclude_input.set_data_from_numpy(np.array([True], dtype=bool))

#     history = history + [
#         {"role": "user", "content": user_input},
#         {"role": "assistant", "content": ""}
#     ]

#     client.async_stream_infer(
#         model_name=MODEL_NAME,
#         inputs=[
#             text_input,
#             stream_input,
#             sampling_input,
#             exclude_input
#         ]
#     )

#     while True:
#         try:
#             token = token_queue.get(timeout=0.4)
#             history[-1]["content"] += token
#             yield history
#         except queue.Empty:
#             break


# # ---------- UI ----------
# with gr.Blocks(title="LLM Playground (Fireworks-style)") as demo:
#     gr.Markdown("## 🔥 LLM Playground (Triton + vLLM)")

#     with gr.Row():
#         with gr.Column(scale=3):
#             chatbot = gr.Chatbot(height=450)
#             user_input = gr.Textbox(
#                 placeholder="Type a message...",
#                 show_label=False
#             )

#         with gr.Column(scale=1):
#             gr.Markdown("### ⚙️ Options")

#             temperature = gr.Slider(0.0, 2.0, value=0.6, label="Temperature")
#             max_tokens = gr.Slider(16, 5000, value=512, step=16, label="Max Tokens")
#             top_p = gr.Slider(0.0, 1.0, value=1.0, label="Top P")
#             top_k = gr.Slider(1, 100, value=40, step=1, label="Top K")
#             presence_penalty = gr.Slider(-2.0, 2.0, value=0.0, label="Presence Penalty")
#             frequency_penalty = gr.Slider(-2.0, 2.0, value=0.0, label="Frequency Penalty")

#     user_input.submit(
#         chat,
#         [
#             user_input,
#             chatbot,
#             temperature,
#             max_tokens,
#             top_p,
#             top_k,
#             presence_penalty,
#             frequency_penalty
#         ],
#         chatbot
#     )

#     user_input.submit(lambda: "", None, user_input)

# demo.launch()
###############################################################################################################################
# import tritonclient.grpc as grpcclient
# import numpy as np
# import json
# import gradio as gr
# import queue
# import time

# # ================= CONFIG =================
# TRITON_URL = "localhost:8001"
# MODEL_NAME = "Llama-3.2-1B-Instruct"
# DEBUG_FUNCTION_CALLING = True

# client = grpcclient.InferenceServerClient(url=TRITON_URL)
# token_queue = queue.Queue()

# # ================= DEBUG LOGGER =================
# def debug_log(title, data=None):
#     if not DEBUG_FUNCTION_CALLING:
#         return
#     print("\n" + "=" * 70)
#     print(f"[DEBUG] {title}")
#     if data is not None:
#         print(data)
#     print("=" * 70)

# # ================= TOOLS =================
# def calculator(operation, a, b):
#     if operation == "add":
#         return a + b
#     if operation == "subtract":
#         return a - b
#     if operation == "multiply":
#         return a * b
#     if operation == "divide":
#         if b == 0:
#             return "Error: division by zero"
#         return a / b
#     return "Unknown operation"

# TOOLS = {
#     "calculator": calculator
# }

# # ================= TOOL PARSER =================
# def try_parse_tool_call(text):
#     debug_log("RAW MODEL OUTPUT", text)

#     try:
#         data = json.loads(text)
#     except Exception as e:
#         debug_log("JSON PARSE FAILED", str(e))
#         return None

#     if "tool_call" not in data:
#         debug_log("NO TOOL CALL FOUND")
#         return None

#     debug_log("TOOL CALL DETECTED", data["tool_call"])
#     return data["tool_call"]

# # ================= STREAM CALLBACK =================
# def callback(result, error):
#     if error:
#         debug_log("STREAM ERROR", error)
#         token_queue.put("")
#         return

#     if result is None:
#         return

#     try:
#         output = result.as_numpy("text_output")
#         if output is not None:
#             token = output[0].decode("utf-8")
#             if token.strip():
#                 token_queue.put(token)
#     except Exception as e:
#         debug_log("CALLBACK ERROR", str(e))

# client.start_stream(callback=callback)

# # ================= CHAT FUNCTION =================
# def chat(
#     user_input,
#     history,
#     temperature,
#     max_tokens,
#     top_p,
#     top_k,
#     presence_penalty,
#     frequency_penalty
# ):
#     token_queue.queue.clear()

#     # ---- Safety clamps ----
#     temperature = max(0.01, float(temperature))
#     top_p = max(0.01, float(top_p))

#     SYSTEM_PROMPT = """
# You are a helpful AI assistant.

# You have access to the following tool:

# calculator(operation, a, b)
# - operation: add | subtract | multiply | divide
# - a: number
# - b: number

# RULES:
# - If a tool is required, respond ONLY in valid JSON.
# - JSON format:
# {
#   "tool_call": {
#     "name": "calculator",
#     "arguments": {
#       "operation": "...",
#       "a": number,
#       "b": number
#     }
#   }
# }
# - If no tool is needed, respond normally in text.
# """

#     prompt = f"""<|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>
# {SYSTEM_PROMPT}
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# {user_input}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """

#     debug_log("USER PROMPT", prompt)

#     # -------- Triton Inputs --------
#     text_input = grpcclient.InferInput("text_input", [1], "BYTES")
#     text_input.set_data_from_numpy(np.array([prompt.encode()], dtype=object))

#     stream_input = grpcclient.InferInput("stream", [1], "BOOL")
#     stream_input.set_data_from_numpy(np.array([True], dtype=bool))

#     sampling_params = {
#         "temperature": temperature,
#         "max_tokens": int(max_tokens),
#         "top_p": top_p,
#         "top_k": int(top_k),
#         "presence_penalty": float(presence_penalty),
#         "frequency_penalty": float(frequency_penalty)
#     }

#     debug_log("SAMPLING PARAMETERS", sampling_params)

#     sampling_input = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
#     sampling_input.set_data_from_numpy(
#         np.array([json.dumps(sampling_params).encode()], dtype=object)
#     )

#     exclude_input = grpcclient.InferInput("exclude_input_in_output", [1], "BOOL")
#     exclude_input.set_data_from_numpy(np.array([True], dtype=bool))

#     history = history + [
#         {"role": "user", "content": user_input},
#         {"role": "assistant", "content": ""}
#     ]

#     # -------- First LLM call --------
#     debug_log("FIRST LLM CALL SENT")

#     client.async_stream_infer(
#         model_name=MODEL_NAME,
#         inputs=[text_input, stream_input, sampling_input, exclude_input]
#     )

#     assistant_text = ""

#     while True:
#         try:
#             token = token_queue.get(timeout=0.4)
#             assistant_text += token
#             history[-1]["content"] += token
#             yield history
#         except queue.Empty:
#             break

#     debug_log("FULL ASSISTANT OUTPUT", assistant_text)

#     # -------- TOOL CALL CHECK --------
#     tool_call = try_parse_tool_call(assistant_text)

#     if not tool_call:
#         debug_log("NO TOOL EXECUTION NEEDED")
#         return

#     tool_name = tool_call.get("name")
#     args = tool_call.get("arguments", {})

#     debug_log("TOOL NAME", tool_name)
#     debug_log("TOOL ARGUMENTS", args)

#     if tool_name not in TOOLS:
#         debug_log("UNKNOWN TOOL", tool_name)
#         return

#     try:
#         result = TOOLS[tool_name](**args)
#         debug_log("TOOL RESULT", result)
#     except Exception as e:
#         result = f"Tool execution failed: {e}"
#         debug_log("TOOL EXECUTION ERROR", str(e))

#     # -------- Second LLM Call --------
#     tool_prompt = f"""<|begin_of_text|>
# <|start_header_id|>assistant<|end_header_id|>
# The tool returned the following result:
# {result}

# Explain this result clearly to the user.
# """

#     debug_log("SECOND PROMPT (TOOL RESULT)", tool_prompt)

#     text_input.set_data_from_numpy(np.array([tool_prompt.encode()], dtype=object))
#     history.append({"role": "assistant", "content": ""})
#     token_queue.queue.clear()

#     client.async_stream_infer(
#         model_name=MODEL_NAME,
#         inputs=[text_input, stream_input, sampling_input, exclude_input]
#     )

#     while True:
#         try:
#             token = token_queue.get(timeout=0.4)
#             history[-1]["content"] += token
#             yield history
#         except queue.Empty:
#             break

#     debug_log("FINAL RESPONSE", history[-1]["content"])

# # ================= UI =================
# with gr.Blocks(title="LLM Playground (Debug Mode)") as demo:
#     gr.Markdown("## 🔥 LLM Playground (Triton + vLLM + Function Calling + Debug)")

#     with gr.Row():
#         with gr.Column(scale=3):
#             chatbot = gr.Chatbot(height=450)
#             user_input = gr.Textbox(placeholder="Type a message...", show_label=False)

#         with gr.Column(scale=1):
#             gr.Markdown("### ⚙️ Options")

#             temperature = gr.Slider(0.01, 2.0, value=0.6, label="Temperature")
#             max_tokens = gr.Slider(16, 5000, value=512, step=16, label="Max Tokens")
#             top_p = gr.Slider(0.01, 1.0, value=1.0, step=0.01, label="Top P")
#             top_k = gr.Slider(1, 100, value=40, step=1, label="Top K")
#             presence_penalty = gr.Slider(-2.0, 2.0, value=0.0, label="Presence Penalty")
#             frequency_penalty = gr.Slider(-2.0, 2.0, value=0.0, label="Frequency Penalty")

#     user_input.submit(
#         chat,
#         [
#             user_input,
#             chatbot,
#             temperature,
#             max_tokens,
#             top_p,
#             top_k,
#             presence_penalty,
#             frequency_penalty
#         ],
#         chatbot
#     )

#     user_input.submit(lambda: "", None, user_input)

# demo.launch()
#############################################################################################################################
import tritonclient.grpc as grpcclient
import numpy as np
import json
import gradio as gr
import queue

# ================= CONFIG =================
TRITON_URL = "localhost:8001"
MODEL_NAME = "Llama-3.2-1B-Instruct"
DEBUG_FUNCTION_CALLING = True

client = grpcclient.InferenceServerClient(url=TRITON_URL)
token_queue = queue.Queue()

# ================= DEBUG LOGGER =================
def debug_log(title, data=None):
    if not DEBUG_FUNCTION_CALLING:
        return
    print("\n" + "=" * 70)
    print(f"[DEBUG] {title}")
    if data is not None:
        print(data)
    print("=" * 70)

# ================= DYNAMIC TOOL REGISTRY =================
DYNAMIC_TOOL_SCHEMAS = []

def add_tool(tool_json_str):
    try:
        tool = json.loads(tool_json_str)
        required_keys = {"name", "description", "parameters"}
        if not required_keys.issubset(tool.keys()):
            return "❌ Invalid tool schema"

        DYNAMIC_TOOL_SCHEMAS.append(tool)
        debug_log("TOOL REGISTERED", tool)
        return f"✅ Tool `{tool['name']}` added"
    except Exception as e:
        return f"❌ JSON Error: {e}"

def build_system_prompt():
    base_prompt = """
You are a helpful AI assistant.

You may have access to external tools.
If a tool is required, respond ONLY in valid JSON.

Format:
{
  "tool_call": {
    "name": "<tool_name>",
    "arguments": { ... }
  }
}

If no tool is needed, respond normally in text.
"""

    if not DYNAMIC_TOOL_SCHEMAS:
        return base_prompt

    tool_descriptions = "\n\n".join(
        [
            f"""
Tool Name: {tool['name']}
Description: {tool.get('description','')}
Parameters:
{json.dumps(tool['parameters'], indent=2)}
"""
            for tool in DYNAMIC_TOOL_SCHEMAS
        ]
    )

    return base_prompt + "\n\nYou have access to the following tools:\n" + tool_descriptions

# ================= TOOL PARSER =================
def extract_json_block(text):
    start = text.find("{")
    if start == -1:
        return None
    brace = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace += 1
        elif text[i] == "}":
            brace -= 1
            if brace == 0:
                return text[start:i+1]
    return None

def try_parse_tool_call(text):
    debug_log("RAW MODEL OUTPUT", text)

    json_block = extract_json_block(text)
    if not json_block:
        debug_log("NO JSON BLOCK FOUND")
        return None

    try:
        data = json.loads(json_block)
    except Exception as e:
        debug_log("JSON PARSE FAILED", str(e))
        return None

    if "tool_call" not in data:
        debug_log("NO TOOL CALL FOUND")
        return None

    debug_log("TOOL CALL DETECTED", data["tool_call"])
    return data["tool_call"]

# ================= STREAM CALLBACK =================
def callback(result, error):
    if error:
        debug_log("STREAM ERROR", error)
        return
    if result:
        out = result.as_numpy("text_output")
        if out is not None:
            token_queue.put(out[0].decode("utf-8"))

client.start_stream(callback=callback)

# ================= CHAT FUNCTION =================
def chat(
    user_input,
    history,
    temperature,
    max_tokens,
    top_p,
    top_k,
    presence_penalty,
    frequency_penalty
):
    token_queue.queue.clear()

    temperature = max(0.01, float(temperature))
    top_p = max(0.01, float(top_p))

    system_prompt = build_system_prompt()

    prompt = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    debug_log("SYSTEM PROMPT", system_prompt)

    text_input = grpcclient.InferInput("text_input", [1], "BYTES")
    text_input.set_data_from_numpy(np.array([prompt.encode()], dtype=object)

    )

    stream_input = grpcclient.InferInput("stream", [1], "BOOL")
    stream_input.set_data_from_numpy(np.array([True], dtype=bool))

    sampling_params = {
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "top_p": top_p,
        "top_k": int(top_k),
        "presence_penalty": float(presence_penalty),
        "frequency_penalty": float(frequency_penalty)
    }

    sampling_input = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
    sampling_input.set_data_from_numpy(
        np.array([json.dumps(sampling_params).encode()], dtype=object)
    )

    exclude_input = grpcclient.InferInput("exclude_input_in_output", [1], "BOOL")
    exclude_input.set_data_from_numpy(np.array([True], dtype=bool))

    history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": ""}
    ]

    client.async_stream_infer(
        model_name=MODEL_NAME,
        inputs=[text_input, stream_input, sampling_input, exclude_input]
    )

    assistant_text = ""

    while True:
        try:
            token = token_queue.get(timeout=0.4)
            assistant_text += token
            history[-1]["content"] += token
            yield history
        except queue.Empty:
            break

    debug_log("FINAL ASSISTANT TEXT", assistant_text)
    try_parse_tool_call(assistant_text)

# ================= UI =================
with gr.Blocks(title="LLM Playground vNext") as demo:
    gr.Markdown("## 🔥 LLM Playground (Dynamic Function Calling)")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            user_input = gr.Textbox(placeholder="Ask something...", show_label=False)

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Options")
            temperature = gr.Slider(0.01, 2.0, value=0.6, label="Temperature")
            max_tokens = gr.Slider(16, 5000, value=512, step=16, label="Max Tokens")
            top_p = gr.Slider(0.01, 1.0, value=1.0, step=0.01, label="Top P")
            top_k = gr.Slider(1, 100, value=40, step=1, label="Top K")
            presence_penalty = gr.Slider(-2.0, 2.0, value=0.0, label="Presence Penalty")
            frequency_penalty = gr.Slider(-2.0, 2.0, value=0.0, label="Frequency Penalty")

    gr.Markdown("### 🔧 Add Function (Tool JSON)")
    tool_json = gr.Textbox(
        lines=10,
        placeholder="Paste Fireworks-style tool JSON here"
    )
    add_tool_btn = gr.Button("➕ Add Tool")
    tool_status = gr.Textbox(label="Tool Status")

    add_tool_btn.click(add_tool, tool_json, tool_status)

    user_input.submit(
        chat,
        [
            user_input,
            chatbot,
            temperature,
            max_tokens,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty
        ],
        chatbot
    )

    user_input.submit(lambda: "", None, user_input)

demo.launch()
