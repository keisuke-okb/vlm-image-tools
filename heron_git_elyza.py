import os
import re
import argparse
import shutil
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from heron.models.git_llm.git_gpt_neox import GitGPTNeoXForCausalLM
from PIL import Image


class HeronGitElyzaBasedVLM:
    def __init__(self, model_name="turing-motors/heron-chat-git-ELYZA-fast-7b-v0", cache_dir="./.cache"):
        # load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GitGPTNeoXForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
            )
        self.model.eval()
        self.model.to(self.device)


    def generate_from_image(self, img_path, instruction):
        # prepare inputs
        image = Image.open(img_path)
        text = f"##human: {instruction}\n##gpt: "

        # do preprocessing
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            truncation=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # set eos token
        eos_token_id_list = [
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.eos_token_id,
        ]

        # do inference
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=256,
                do_sample=False,
                temperature=0.,
                eos_token_id=eos_token_id_list,
                )

        # result
        output = self.processor.tokenizer.batch_decode(out)[0]
        return output
    

    def run_tools(self, input_dir, output_dir, task, input, save_text=False, rename=False, classify=False):
        images = os.listdir(input_dir)
        os.makedirs(output_dir, exist_ok=True)

        for image in tqdm(images):
            output = self.generate_from_image(os.path.join(input_dir, image), task, input)
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
    p.add_argument('--instruction', default="この画像の面白い点は何ですか?")
    p.add_argument('--save_text', action="store_true")
    p.add_argument('--rename', action="store_true")
    p.add_argument('--classify', action="store_true")
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = HeronGitElyzaBasedVLM(args.model_name, args.cache_dir)
    model.run_tools(
        args.input_dir,
        args.output_dir,
        args.instruction,
        args.save_text,
        args.rename,
        args.classify,
    )