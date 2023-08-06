# 以下是一个简单的样例：

# ```python
import json
import tornado.ioloop
import tornado.web
from transformers import AutoTokenizer, AutoModel
import torch.cuda


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=CACHE_DIR)

    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=CACHE_DIR)
    if torch.cuda.is_available():
        model = model.half().cuda()
    else:
        model = model.float()
    model.eval()
    return tokenizer, model


class ShanghaiDialectTTSHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Content-Type', 'application/octet-stream')

    def post(self):
        try:
            request_data = json.loads(self.request.body)
            sentence = request_data.get("sentence")
            if not sentence:
                raise ValueError("Missing or invalid 'sentence' parameter")

            output_file = init_model(sentence)

            with open(output_file, 'rb') as f:
                self.write(f.read())

            self.finish()

        except ValueError as e:
            self.set_status(400)
            self.write({"error": str(e)})
            self.finish()

        except Exception as e:
            self.set_status(500)
            self.write({"error": "Internal server error"})
            self.finish()

def make_app():
    return tornado.web.Application([
        (r"/api/text2speech", ShanghaiDialectTTSHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(19941)  # 监听 19941 端口
    tornado.ioloop.IOLoop.current().start()
# ```