import os
import re
import argparse
import shutil
import torch
from tqdm import tqdm
from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
from transformers import LlamaTokenizer
from PIL import Image


class HeronBlipStableLMBasedVLM:
    def __init__(self, model_name="turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0", cache_dir="./.cache", half=True):
        # load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VideoBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir,
        )
        self.processor = VideoBlipProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            cache_dir=cache_dir
            )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1",
            additional_special_tokens=['▁▁'],
            cache_dir=cache_dir,
            )
        self.processor.tokenizer = self.tokenizer
        if half:
            self.model.half()
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
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device, torch.float16)

        # set eos token
        eos_token_id_list = [
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.eos_token_id,
            int(self.tokenizer.convert_tokens_to_ids("##"))
        ]

        # do inference
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=256,
                do_sample=False,
                temperature=0.,
                eos_token_id=eos_token_id_list,
                no_repeat_ngram_size=2
                )

        # result
        output = self.processor.tokenizer.batch_decode(out)
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
                    output_path
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
    model = HeronBlipStableLMBasedVLM(args.model_name, args.cache_dir)
    model.run_tools(
        args.input_dir,
        args.output_dir,
        args.instruction,
        args.save_text,
        args.rename,
        args.classify,
    )