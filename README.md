# vlm-image-tools
Image classification and captioning tools using VLMs (Stable VLM, Heron)

## Features
Using VLMs(Visual Language Models), many image files can be processed at once:

Output types(multiple selections supported):
- `--save_text`: Save image file name and its corresponding result to text file.
- `--rename`: Rename files according to model output
- `--classify`: Classify image files into several folders according to model output

Supported tasks:
- Stable VLM: "caption", "tag", "vqa"
- Heron: "instruction"

## Usage

### 1. Commandline

```powershell
cd vlm-image-tools
python stable_vlm.py \
  --model_name "stabilityai/japanese-stable-vlm" \
  --input_dir "path/to/input_images" \
  --output_dir "path/to/input_images" \
  --task "caption" \
  --save_text \
  --rename \
  --classify
```

### 2. Jupyter Notebook

```python
from stable_vlm import StableVLM
model = StableVLM("Z:\models\japanese-stable-vlm")

model.run_tools(
    input_dir="path/to/images_input",
    output_dir="path/to/images_output",
    task="caption",
    input="",
    do_sample=False,
    save_text=True,
    rename=True,
    classify=False,
)
```

### Demo

Task:
Grouping below animal images into folders by the animal's name.
Images: https://pixabay.com/
![image](https://github.com/keisuke-okb/vlm-image-tools/assets/70097451/f3e4267b-ef23-42a8-a030-d9c94eae5b99)

```python
model.run_tools(
    input_dir=r"Z:\temp\images\animals",
    output_dir=r"Z:\temp\images\animals_vlm",
    task="vqa",
    input="この画像に写っている動物は何ですか？",
    do_sample=False,
    save_text=False,
    rename=False,
    classify=True,
)
```

Result:
One folder name is "この写っている画像を下に、質問に答えてください。" for the lion image.
![image](https://github.com/keisuke-okb/vlm-image-tools/assets/70097451/1fb0c853-14fb-4748-9a6e-004adacb50e3)

# References

Japanese Stable VLM
https://ja.stability.ai/blog/japanese-stable-vlm

Heron
https://github.com/turingmotors/heron






