# 0 本地部署

1. 下载代码
   ```bash
   git clone https://github.com/PKU-Alignment/align-anything
   ```
   安装依赖
   ```bash
   conda create -n align-anything python=3.11
   conda activate align-anything
   pip install -e .
   ```
2. 下载模型
   ```bash
   huggingface-cli download --resume-download PKU-Alignment/Beaver-0.5B-Instruct --local-dir LOCAL_DIR
   
   ```
   ```text
   Fetching 10 files:   0%|                                                                                                                                                                                  | 0/10 [00:00<?, ?it/s]Downloading '.gitattributes' to 'LOCAL_DIR/.cache/huggingface/download/.gitattributes.52373fe24473b1aa44333d318f578ae6bf04b49b.incomplete'
   Downloading 'README.md' to 'LOCAL_DIR/.cache/huggingface/download/README.md.7b95401dc46245ac339fc25059d4a56d90b4cde5.incomplete'
   .gitattributes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.57k/1.57k [00:00<00:00, 5.99MB/s]
   Download complete. Moving file to LOCAL_DIR/.gitattributes                                                                                                                                           | 0.00/1.57k [00:00<?, ?B/s]
   README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.0/31.0 [00:00<00:00, 126kB/s]
   Download complete. Moving file to LOCAL_DIR/README.md                                                                                                                                                 | 0.00/31.0 [00:00<?, ?B/s]
   Fetching 10 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 10.45it/s]
   /data/user1/homework/llm/align-anything/LOCAL_DIR

   ```
3. 下载数据
   ```bash
   huggingface-cli download --repo-type dataset --resume-download PKU-Alignment/PKU-SafeRLHF-single-dimension --local-dir LOCAL_DIR --local-dir-use-symlinks False
   ```
   ```text
   Fetching 9 files:   0%|                                                                                                                                                                                    | 0/9 [00:00<?, ?it/s]Downloading 'README.md' to 'LOCAL_DIR/.cache/huggingface/download/README.md.e32c675367b903ed8a047a391db9fad88e3bb7c1.incomplete'
   README.md: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.71k/1.71k [00:00<00:00, 6.45MB/s]
   Download complete. Moving file to LOCAL_DIR/README.md                                                                                                                                                | 0.00/1.71k [00:00<?, ?B/s]
   Downloading '.gitattributes' to 'LOCAL_DIR/.cache/huggingface/download/.gitattributes.fe9ab76f5638b454a47bbfc31fe432b2546d6e7c.incomplete'
   .gitattributes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.84k/2.84k [00:00<00:00, 18.2MB/s]
   Download complete. Moving file to LOCAL_DIR/.gitattributes                                                                                                                                           | 0.00/2.84k [00:00<?, ?B/s]
   Fetching 9 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:01<00:00,  8.68it/s]
   /data/user1/homework/llm/align-anything/LOCAL_DIR

   ```
   

# 1 reward model implementation

## 1.1 偏好数据集键值转换


```python
@register_template('HOMEWORK')
class HOMEWORK:
    def __init__(self) -> None:
        # reference class PKUSafeRLHF(Template):
        self.system_prompt: str = 'BEGINNING OF CONVERSATION: '
        self.user_prompt: str = 'USER: {input} '
        self.assistant_prompt: str = 'ASSISTANT:{output}'
        self.split_token: str = 'ASSISTANT:'
        self.separator: str = ''

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        metrics = raw_sample['better_response_id']
        better_response = raw_sample[f'response_{int(metrics)}']
        worse_response = raw_sample[f'response_{1 - int(metrics)}']
        prompt = raw_sample['prompt']

        formatted_better_output = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=worse_response)}'
        )

        return {
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
        }


```
