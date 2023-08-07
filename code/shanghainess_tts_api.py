from typing import Optional, Awaitable
from transformers import AutoTokenizer, AutoModel
import torch.cuda
import tornado.ioloop
import tornado.web
import json
import logging
import os
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence

CACHE_DIR = "cache"

def init_model():
    print("intmodel--->")
    hps_ms = utils.get_hparams_from_file('model/config.json')
    n_speakers = hps_ms.data.n_speakers
    n_symbols = len(hps_ms.symbols)
    print("n_symbols",n_symbols)
    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint('model/model.pth', net_g_ms)
    if torch.cuda.is_available():
        net_g_ms = net_g_ms.cuda(device="cuda")

    net_g_ms.eval()
    print("net_g_ms",net_g_ms)
    return net_g_ms


def text2speech(username: str, self=None):
    # filename = t2s(xxx)
    # self.write({"filename": "shanghainess-tts.wav"})
    return "shanghainess-tts.wav"


class MainHandler(tornado.web.RequestHandler):
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def set_default_headers(self):
        # self.set_header('Content-Type', 'text/plain')
        self.set_header('Content-Type', 'text/event-stream')

    def get(self, sentence=None):
        self.write("Hello, World!")

    def post(self):
        # sentence = self.get_argument('sentence', default=None)
        # data = self.request.body
        body = json.loads(self.request.body)
        sentence = body.get('sentence', None)
        if sentence:
            print("sentence--->",sentence)
            # filename = text2speech(username)
            filename = init_model(sentence)
            self.write(f"Username: {filename}")
        else:
            self.write("Missing username")


class GenerateHandler(tornado.web.RequestHandler):
    print("GenerateHandler--->")
    model = None
    hps_ms = None

    def initialize(self, model):
        print("model---initialize>")
        self.model = model
        self.hps_ms = utils.get_hparams_from_file('model/config.json')
        pass

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def set_default_headers(self):
        self.set_header('Content-Type', 'application/json')

    def get(self):
        self.write("welcome to haydn Shanghai dialect TTS server.")

    def post(self):
        try:
            sentence, noise_scale, noise_scale_w, length_scale = self.parse_arguments(str(self.request.body, "utf-8"))
            print("sentence--->",sentence)
            sentence = sentence.replace('\n', '')

            sequence = self.sentence_to_sequence(sentence)
            audio_data = audio_data / numpy.abs(audio_data).max()
            audio_data = audio_data * 32767
            audio_data = audio_data.astype(numpy.int16)

            sound = pydub.AudioSegment(audio_data, frame_rate=self.hps_ms.data.sampling_rate, sample_width=audio_data.dtype.itemsize, channels=1)
            sound.export(export_filename, format=AUDIO_FORMAT)
            self.write_response(export_url=export_url)

        except Exception as e:
            self.write_response(code=500, message=str(e))
            logging.error("{} {}".format(str(e), str(self.request.remote_ip)))

    def sentence_to_sequence(self, sentence):
        print("sentence_to_sequence--->")
        sequence = text_to_sequence(sentence, self.hps_ms.symbols, self.hps_ms.data.text_cleaners)
        if self.hps_ms.data.add_blank:
            sequence = commons.intersperse(sequence, 0)
        sequence = torch.LongTensor(sequence)
        return sequence

    def write_response(self, code=0, message="success", export_url=None):
        print("write_response------>")
        if code != 0:
            self.write(json.dumps({
                "code": code, "message": message
            }))
        else:
            self.write(json.dumps({
                "code": code, "message": message, "data": {
                    "filename": export_url
                }
            }))
        self.flush()


# curl localhost:8081/
def main():
    print("main--->")
    # parser.add_argument('--device', default="cuda:0")
    # os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    model = init_model()
    print("init--->")
    application = tornado.web.Application(
        handlers=[
            ("/api/shanghai-tts/" + "/generate", GenerateHandler, {"model": model})
        ],
        static_path="/output",
        static_url_prefix="/api/shanghai-tts/" + "/audio/",
        debug=False
    )
    print("api")
    print("static_url_prefix----------"，static_url_prefix)
    application.listen(8083)

    logging.info("URL".format(8082))

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
