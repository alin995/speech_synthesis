


# API
```bash
curl -XPOST http://172.30.35.51/haydn-assistant/api/agent/text2speech/generate \
-H "token: i-love-thundax" \
-H "Content-Type: application/json" \
-d'{
    "sentence": "写一篇800字的新闻，要点：樱花节、顾村公园"
}'

```
```bash
curl -XPOST http://localhost:8082/ \
-H "token: i-love-thundax" \
-H "Content-Type: application/json" \
-d'{
    "sentence": "写一篇800字的新闻"
}'

```


```bash
curl -X POST http://localhost:8082/ \
-H "Content-Type: application/json" \
-d '{
    "sentence": "文本转语音"
}'

```
```bash
curl -XPOST http://localhost:8082/api/shanghai-tts/generate \
-d'{
"sentence": "写一篇800字的新闻，要点：樱花节、顾村公园"
}'
```

curl -XPOST http://127.0.0.1:8082/api/shanghai-tts/generate \
-d'{
"sentence": "写一篇800字的新闻，要点：樱花节、顾村公园"
}'