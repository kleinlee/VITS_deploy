# VITS_deploy

本项目完全为MiniMates项目的自定义语音模块服务。 

https://github.com/kleinlee/MiniMates 完全面向用户的数字人！

请先参照https://github.com/Plachtaa/VITS-fast-fine-tuning 完成模型训练。

注意，本脚本目前只支持纯中文的语音模型。如需要进行英文或日语等的模型部署，请自行魔改。

本脚本来源于https://github.com/Plachtaa/VITS-fast-fine-tuning/issues/518 做了轻度修改以满足MiniMates的需要。非常感谢新一代kaldi团队的贡献！

## Usage

将这两个脚本放置于VITS-fast-fine-tuning项目内

先执行generate-lexicon-zh-hf-fanchen-models.py  转换一些简单的配置文件

再执行export-onnx-zh-hf-fanchen-models.py  将VITS模型转换为onnx模型

最后，请收集生成的文件，保证格式为：
```bash
|--/sherpa-onnx-vits-model
|  |--/dict
|  |--/finetune_speaker.json
|  |--/model.onnx
|  |--/lexicon.txt
|  |--/tokens.txt
```
