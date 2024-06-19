import os
import re
from io import BytesIO
from PIL import Image
import telebot
from transformers import GPTNeoForCausalLM, AutoTokenizer
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# إعداد البيئة
TOKEN = os.getenv('TELEGRAM_TOKEN')
AZURE_SUBSCRIPTION_KEY = os.getenv('AZURE_SUBSCRIPTION_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
HUGGING_FACE_API_ENDPOINT = os.getenv('HUGGING_FACE_API_ENDPOINT')

# إعداد Azure OCR
computervision_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_SUBSCRIPTION_KEY))

# تحميل النموذج والمحول
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

bot = telebot.TeleBot(TOKEN)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9ء-ي\s]', '', text)  # إزالة الحروف غير الضرورية
    return text.strip()

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, 'مرحبًا! أرسل لي صورة تحتوي على الأسئلة أو التمارين.')

@bot.message_handler(content_types=['photo'])
def handle_image(message):
    photo_file = bot.get_file(message.photo[-1].file_id)
    photo_bytes = BytesIO(bot.download_file(photo_file.file_path))
    image = Image.open(photo_bytes)
    
    # تحويل الصورة إلى تنسيق يمكن إرساله إلى Azure
    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    # استخراج النص من الصورة باستخدام Azure OCR
    ocr_results = computervision_client.read_in_stream(BytesIO(image_bytes), raw=True)
    operation_location = ocr_results.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    # انتظار النتائج
    while True:
        result = computervision_client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break

    if result.status == OperationStatusCodes.succeeded:
        extracted_text = ""
        for text_result in result.analyze_result.read_results:
            for line in text_result.lines:
                extracted_text += line.text + " "
    else:
        bot.reply_to(message, 'حدث خطأ أثناء قراءة النص من الصورة.')
        return

    cleaned_text = clean_text(extracted_text)

    # استخدام النموذج للإجابة على الأسئلة
    inputs = tokenizer.encode(cleaned_text + tokenizer.eos_token, return_tensors='pt')
    reply_ids = model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    bot.reply_to(message, reply)

if __name__ == '__main__':
    bot.polling()
