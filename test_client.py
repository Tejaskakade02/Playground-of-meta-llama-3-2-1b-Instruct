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
import tritonclient.grpc as grpcclient
import numpy as np
import json
import gradio as gr
import queue

# ---------------- CONFIG ----------------
TRITON_URL = "localhost:8001"
MODEL_NAME = "Llama-3.2-1B-Instruct"

client = grpcclient.InferenceServerClient(url=TRITON_URL)
token_queue = queue.Queue()

# ---------- STREAM CALLBACK ----------
def callback(result, error):
    if error:
        token_queue.put(f"\n❌ Error: {error}")
        return

    if result is None:
        return

    try:
        output = result.as_numpy("text_output")
        if output is not None:
            token = output[0].decode("utf-8")
            if token.strip():
                token_queue.put(token)
    except Exception:
        pass


client.start_stream(callback=callback)

# ---------- CHAT FUNCTION ----------
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

    prompt = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful, polite AI assistant.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    # -------- Triton Inputs --------
    text_input = grpcclient.InferInput("text_input", [1], "BYTES")
    text_input.set_data_from_numpy(np.array([prompt.encode()], dtype=object))

    stream_input = grpcclient.InferInput("stream", [1], "BOOL")
    stream_input.set_data_from_numpy(np.array([True], dtype=bool))

    sampling_params = {
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "top_p": top_p,
        "top_k": int(top_k),
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty
    }

    sampling_input = grpcclient.InferInput(
        "sampling_parameters", [1], "BYTES"
    )
    sampling_input.set_data_from_numpy(
        np.array([json.dumps(sampling_params).encode()], dtype=object)
    )

    exclude_input = grpcclient.InferInput(
        "exclude_input_in_output", [1], "BOOL"
    )
    exclude_input.set_data_from_numpy(np.array([True], dtype=bool))

    history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": ""}
    ]

    client.async_stream_infer(
        model_name=MODEL_NAME,
        inputs=[
            text_input,
            stream_input,
            sampling_input,
            exclude_input
        ]
    )

    while True:
        try:
            token = token_queue.get(timeout=0.4)
            history[-1]["content"] += token
            yield history
        except queue.Empty:
            break


# ---------- UI ----------
with gr.Blocks(title="LLM Playground (Fireworks-style)") as demo:
    gr.Markdown("## 🔥 LLM Playground (Triton + vLLM)")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            user_input = gr.Textbox(
                placeholder="Type a message...",
                show_label=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Options")

            temperature = gr.Slider(0.0, 2.0, value=0.6, label="Temperature")
            max_tokens = gr.Slider(16, 5000, value=512, step=16, label="Max Tokens")
            top_p = gr.Slider(0.0, 1.0, value=1.0, label="Top P")
            top_k = gr.Slider(1, 100, value=40, step=1, label="Top K")
            presence_penalty = gr.Slider(-2.0, 2.0, value=0.0, label="Presence Penalty")
            frequency_penalty = gr.Slider(-2.0, 2.0, value=0.0, label="Frequency Penalty")

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
