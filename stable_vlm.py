import os
import re
import argparse
import shutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor
from PIL import Image

# helper function to format input prompts
TASK2INSTRUCTION = {
    "caption": "画像を詳細に述べてください。",
    "tag": "与えられた単語を使って、画像を詳細に述べてください。",
    "vqa": "与えられた画像を下に、質問に答えてください。",
}

class StableVLM:
    def __init__(self, model_name="stabilityai/japanese-stable-vlm", cache_dir="./.cache", half=True):
        # load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            )
        if half:
            self.model.half()
        self.model.to(self.device)
    

    def build_prompt(self, task="caption", input=None, sep="\n\n### "):
        assert (
            task in TASK2INSTRUCTION
        ), f"Please choose from {list(TASK2INSTRUCTION.keys())}"
        if task in ["tag", "vqa"]:
            assert input is not None, "Please fill in `input`!"
            if task == "tag" and isinstance(input, list):
                input = "、".join(input)
        else:
            assert input is None, f"`{task}` mode doesn't support to input questions"
        sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
        p = sys_msg
        roles = ["指示", "応答"]
        instruction = TASK2INSTRUCTION[task]
        msgs = [": \n" + instruction, ": \n"]
        if input:
            roles.insert(1, "入力")
            msgs.insert(1, ": \n" + input)
        for role, msg in zip(roles, msgs):
            p += sep + role + msg
        return p


    def generate_from_image(self, img_path, task, input, do_sample=False):
        # prepare inputs
        image = Image.open(img_path).convert("RGB")
        if task == "caption":
            prompt = self.build_prompt(task=task)
        elif task == "tag":
            prompt = self.build_prompt(task=task, input=input.split(","))
        elif task == "vqa":
            prompt = self.build_prompt(task=task, input=input)
        else:
            raise NotImplementedError()
        
        inputs = self.processor(images=image, return_tensors="pt")
        text_encoding = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs.update(text_encoding)

        # generate
        outputs = self.model.generate(
            **inputs.to(self.device, dtype=self.model.dtype),
            do_sample=do_sample,
            num_beams=5,
            max_new_tokens=512,
            min_length=1,
            repetition_penalty=1.5,
        )

        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text
    

    def run_tools(self, input_dir, output_dir, task, input, do_sample=False, save_text=False, rename=False, classify=False):
        images = os.listdir(input_dir)
        os.makedirs(output_dir, exist_ok=True)

        for image in tqdm(images):
            output = self.generate_from_image(os.path.join(input_dir, image), task, input, do_sample)
            if save_text:
                with open(os.path.join(output_dir, "result.txt"), "a", encoding="UTF-8") as f:
                    print(f"{image}, {output}", file=f)

            output = re.sub(r'[\\|/|:|?|.|"|<|>|\|]', '_', output)
            output = output[:30]

            if rename:
                idx = 1
                output_path = os.path.join(output_dir, output + f"_{idx:04d}" + os.path.splitext(image)[1])
                while os.path.exists(output_path):
                    idx += 1
                    output_path = os.path.join(output_dir, output + f"_{idx:04d}" + os.path.splitext(image)[1])

                shutil.copyfile(
                    os.path.join(input_dir, image),
                    os.path.join(output_dir, output + os.path.splitext(image)[1])
                )
            
            elif classify:
                output_dir_sub = os.path.join(output_dir, output)
                os.makedirs(output_dir_sub, exist_ok=True)
                shutil.copyfile(
                    os.path.join(input_dir, image),
                    os.path.join(output_dir_sub, image)
                )


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', default="stabilityai/japanese-stable-vlm")
    p.add_argument('--cache_dir', default="./.cache")
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--task', choices=["caption", "tag", "vqa"])
    p.add_argument('--input', default="画像には何が写っていますか？")
    p.add_argument('--do_sample', action="store_true")
    p.add_argument('--save_text', action="store_true")
    p.add_argument('--rename', action="store_true")
    p.add_argument('--classify', action="store_true")
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = StableVLM(args.model_name, args.cache_dir)
    model.run_tools(
        args.input_dir,
        args.output_dir,
        args.task,
        args.input,
        args.do_sample,
        args.save_text,
        args.rename,
        args.classify,
    )