import os
import torch
import torchaudio
from clearvoice import ClearVoice

# ----------------------------
# تنظیمات
# ----------------------------
INPUT_clearvoice = "test_robot/robot_test4.mp3"
OUTPUT_DIR = "outputs_clearvoice"
SAMPLE_RATE = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# بارگذاری pipeline
# ----------------------------
model = ClearVoice(
    task="speech_separation",
    model_names=['MossFormer2_SS_16K']
    
)

print("ClearVoice loaded.")
#device="cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------
# اجرای separation
# ----------------------------
# مدل پیش‌فرض: MossFormer2_SS_16K
#separated_audios = model.process(INPUT_WAV)

model(
    input_path=INPUT_clearvoice,
    online_write=True,
    output_path=OUTPUT_DIR
)

# print(f"Detected {len(separated_audios)} speakers.")

# # ----------------------------
# # ذخیره خروجی‌ها
# # ----------------------------
# for i, audio in enumerate(separated_audios):
#     output_path = os.path.join(OUTPUT_DIR, f"speaker_{i+1}.wav")

#     # audio: Tensor [T]
#     torchaudio.save(
#         output_path,
#         audio.unsqueeze(0),
#         SAMPLE_RATE
#     )

#     print(f"Saved: {output_path}")
