import torch
import librosa
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
import numpy as np
from mel_processing import spectrogram_torch
import gradio as gr
from text.cleaners import shanghainese_cleaners

DEFAULT_TEXT='阿拉小人天天辣辣白相，书一眼也勿看，拿我急煞脱了。侬讲是𠲎？'


def clean_text(text,ipa_input):
    if ipa_input:
        return shanghainese_cleaners(text)
    return text


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def speech_synthesize(text, cleaned, length_scale):
    text=text.replace('\n','')
    print(text)
    stn_tst = get_text(text, hps_ms, cleaned)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([0])
        print("x_tst: " + str(x_tst))
        print("x_tst: " + str(x_tst.device))
        print("x_tst_lengths: " + str(x_tst_lengths))
        print("x_tst_lengths: " + str(x_tst_lengths.device))
        print("sid: " + str(sid))
        print("sid: " + str(sid.device))
        x_tst= x_tst.to("cuda")
        x_tst_lengths=x_tst_lengths.to("cuda")
        sid=sid.to("cuda")
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
    return (hps_ms.data.sampling_rate, audio)


if __name__=='__main__':
    hps_ms = utils.get_hparams_from_file('model/config.json')
    n_speakers = hps_ms.data.n_speakers
    n_symbols = len(hps_ms.symbols)
    speakers = hps_ms.speakers

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint('model/model.pth', net_g_ms)
    net_g_ms.to("cuda")
    with gr.Blocks() as app:
        gr.Markdown('# Shanghainese Text to Speech\n'
        '![visitor badge](https://visitor-badge.glitch.me/badge?page_id=cjangcjengh.shanghainese-tts)')
        gr.Markdown('<center><big><b>See <a href="https://huggingface.co/spaces/CjangCjengh/Shanghainese-TTS/raw/main/shanghainese_script.txt">EXAMPLES</a> on Shanghainese script</b></big></center>')
        text_input = gr.TextArea(label='Text', placeholder='Type your text here',value=DEFAULT_TEXT)
        cleaned_text=gr.Checkbox(label='IPA Input',defaulu=True)
        length_scale=gr.Slider(0.5,2,1,step=0.1,label='Speaking Speed',interactive=True)
        tts_button = gr.Button('Synthesize')
        audio_output = gr.Audio(label='Speech Synthesized')
        cleaned_text.change(clean_text,[text_input,cleaned_text],[text_input])
        tts_button.click(speech_synthesize,[text_input,cleaned_text,length_scale],[audio_output])
        gr.Markdown('## Based on\n'
        '- [https://github.com/jaywalnut310/vits](https://github.com/jaywalnut310/vits)\n\n'
        '## Dataset\n'
        '- [http://shh.dict.cn/](http://shh.dict.cn/)\n\n'
        '## Lexicon\n'
        '- [https://www.wugniu.com/](https://www.wugniu.com/)\n\n'
        '- [https://github.com/MaigoAkisame/MCPDict](https://github.com/MaigoAkisame/MCPDict)\n\n'
        '- [https://github.com/edward-martyr/rime-yahwe_zaonhe](https://github.com/edward-martyr/rime-yahwe_zaonhe)')

    app.launch(server_name="0.0.0.0", server_port=18081)
