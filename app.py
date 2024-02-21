import asyncio
import datetime
import logging
import os
import time
import traceback

import edge_tts
import gradio as gr
import librosa
import torch
from fairseq import checkpoint_utils

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

limitation = os.getenv("SYSTEM") == "spaces"

config = Config()

# Edge TTS
edge_output_filename = "edge_output.mp3"
tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]

# RVC models
model_root = "weights"
models = [d for d in os.listdir(model_root) if os.path.isdir(f"{model_root}/{d}")]
models.sort()


def model_data(model_name):
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_path = [
        f"{model_root}/{model_name}/{f}"
        for f in os.listdir(f"{model_root}/{model_name}")
        if f.endswith(".pth")
    ][0]
    print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    # n_spk = cpt["config"][-3]

    index_files = [
        f"{model_root}/{model_name}/{f}"
        for f in os.listdir(f"{model_root}/{model_name}")
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        print(f"Index file found: {index_file}")

    return tgt_sr, net_g, vc, version, index_file, if_f0


def load_hubert():
    # global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


def tts(
    model_name,
    speed,
    tts_text,
    tts_voice,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    custom_audio,
    approval,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
):
    if os.getenv("APPROVAL") == approval:
        approval = True
    else:
        approval = False
    if not approval:
        return "Approval failed", None, None

    print("------------------")
    print(datetime.datetime.now())
    print("tts_text:")
    print(tts_text)
    print(f"tts_voice: {tts_voice}, speed: {speed}")
    print(f"Model name: {model_name}")
    print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")
    try:
        if limitation and len(tts_text) > 300:
            print("Error: Text too long")
            return (
                f"Text characters should be at most 280 in this huggingface space, but got {len(tts_text)} characters.",
                None,
                None,
            )
        t0 = time.time()
        if speed >= 0:
            speed_str = f"+{speed}%"
        else:
            speed_str = f"{speed}%"

        file_path = edge_output_filename

        if custom_audio is not None:
            file_path = custom_audio
        else:
            asyncio.run(
                edge_tts.Communicate(
                    tts_text, "-".join(tts_voice.split("-")[:-1]), rate=speed_str
                ).save(file_path)
            )
        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        if limitation and duration >= 90:
            print("Error: Audio too long")
            return (
                f"Audio should be less than 90 seconds in this huggingface space, but got {duration}s.",
                file_path,
                None,
            )
        f0_up_key = int(f0_up_key)

        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            file_path,
            times,
            f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        return (
            info,
            file_path,
            (tgt_sr, audio_opt),
        )
    except EOFError:
        info = (
            "It seems that the edge-tts output is not valid. "
            "This may occur when the input text and the speaker do not match. "
            "For example, maybe you entered Japanese (without alphabets) text but chose non-Japanese speaker?"
        )
        print(info)
        return info, None, None
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None


print("Loading hubert model...")
hubert_model = load_hubert()
print("Hubert model loaded.")

print("Loading rmvpe model...")
rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)
print("rmvpe model loaded.")

initial_md = """
#
Input characters are limited to 280 characters, and the speech audio is limited to 20 seconds in this 🤗 space.

[Visit this GitHub repo](https://github.com/litagin02/rvc-tts-webui) for running locally with your models and GPU!
"""

app = gr.Blocks()
with app:
    gr.Markdown(initial_md)
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(
                label="Model",
                choices=models,
                value=models[0],
            )
            f0_key_up = gr.Number(
                label="Tune (+12 = 1 octave up from edge-tts, the best value depends on the models and speakers)",
                value=2,
            )
            approval = gr.Textbox(
                label="Approval token",
                value="",
            )
        with gr.Column():
            f0_method = gr.Radio(
                label="Pitch extraction method (pm: very fast, low quality, rmvpe: a little slow, high quality)",
                choices=["pm", "rmvpe"],  # harvest and crepe is too slow
                value="rmvpe",
                interactive=True,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label="Index rate",
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label="Protect",
                value=0.33,
                step=0.01,
                interactive=True,
            )
    with gr.Row():
        with gr.Column():
            tts_voice = gr.Dropdown(
                label="Edge-tts speaker (format: language-Country-Name-Gender), make sure the gender matches the model",
                choices=tts_voices,
                allow_custom_value=False,
                value="ja-JP-NanamiNeural-Female",
            )
            speed = gr.Slider(
                minimum=-100,
                maximum=100,
                label="Speech speed (%)",
                value=0,
                step=10,
                interactive=True,
            )
            tts_text = gr.Textbox(label="Input Text", value="Input text here")
            custom_audio = gr.Audio(label="Custom audio", type="filepath", sources=["microphone", "upload"])
        with gr.Column():
            but0 = gr.Button("Convert", variant="primary")
            edge_tts_output = gr.Audio(label="Reference voice", type="filepath")
            tts_output = gr.Audio(label="Result")
            info_text = gr.Textbox(label="Output info")
        but0.click(
            tts,
            [
                model_name,
                speed,
                tts_text,
                tts_voice,
                f0_key_up,
                f0_method,
                index_rate,
                protect0,
                custom_audio,
                approval,
            ],
            [info_text, edge_tts_output, tts_output],
        )
    with gr.Row():
        examples = gr.Examples(
            examples_per_page=100,
            examples=[
                ["これは日本語テキストから音声への変換デモです。", "ja-JP-NanamiNeural-Female"],
                [
                    "This is an English text to speech conversation demo.",
                    "en-US-AriaNeural-Female",
                ],
                ["这是一个中文文本到语音的转换演示。", "zh-CN-XiaoxiaoNeural-Female"],
                [
                    "Il s'agit d'une démo de conversion du texte français à la parole.",
                    "fr-FR-DeniseNeural-Female",
                ],
                [
                    "Dies ist eine Demo zur Umwandlung von Deutsch in Sprache.",
                    "de-DE-AmalaNeural-Female",
                ]
            ],
            inputs=[tts_text, tts_voice],
        )


app.launch()
