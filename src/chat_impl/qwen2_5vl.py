from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from src.chat_impl.base_generator import BaseGenerator

class Qwen2_5VL(BaseGenerator):

    def load_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda" if self.gpu else "cpu")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        return model, processor
    
    def chat(self, history) -> str:
        model, processor = self.model
        device = 'cuda' if self.gpu else 'cpu'

        history = [{"role": "system", "content": self.system_prompt}, *history]

        # Preparation for inference
        text = processor.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(history)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_ids=inputs.input_ids.to(device)
        attention_mask=inputs.attention_mask.to(device)
        pixel_values=inputs.pixel_values.to(device)
        image_grid_thw=inputs.image_grid_thw.to(device)

        # Inference
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                       pixel_values=pixel_values, image_grid_thw=image_grid_thw, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    