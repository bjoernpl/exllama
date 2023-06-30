import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import subprocess
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel
from translatepy.translators.google import GoogleTranslate
import sounddevice as sd
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa

from playsound import playsound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device", default="auto", help="device to user for CTranslate2 inference",
                        choices=["auto", "cuda","cpu"])
    parser.add_argument("--compute_type", default="auto", help="Type of quantization to use",
                        choices=["auto", "int8", "int8_floatt16", "float16", "int16", "float32"])
    parser.add_argument("--translation_lang", default='German',
                        help="Which language should we translate into." , type=str)
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--threads", default=0,
                        help="number of threads used for CPU inference", type=int)
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=3,
                        help="How real time the recording is in seconds.", type=float)

    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)

    args = parser.parse_args()

    phrase_time = None
    last_sample = bytes()
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    if args.model == "large":
        args.model = "large-v2"

    model = args.model
    if args.model != "large-v2" and not args.non_english:
        model = model 

    translation_lang = args.translation_lang
    device = args.device
    compute_type = args.compute_type if device != "cpu" else "int8"

    nltk.download('punkt')
    audio_model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=args.threads)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    play_obj = None
    
    MIN_TRANSFORMERS_VERSION = '4.25.1'

    def ask_hugging_face(question):
        assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1") 
        response = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
        response = response.to('cuda:0')

        prompt = f'<human>: {question}\n<bot>:'
        inputs = tokenizer(prompt, return_tensors='pt').to(response.device)
        input_length = inputs.input_ids.shape[1]

        output = response.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            return_dict_in_generate=True
        )

        answer = output.sequences[0, input_length:]
        answer = tokenizer.decode(answer)

        while len(answer) < 5:
            answer += ' '

        model_name = 'tts_models/de/thorsten/tacotron2-DDC'
        tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
        tts.tts_to_file(text=answer, file_path="output.wav")

        wav_obj = sa.WaveObject.from_wave_file("output.wav")
        nonlocal play_obj
        play_obj = wav_obj.play()

        return answer

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                phrase_time = now 

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data 

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data()) 

                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read()) 

                text = ""

                segments, info = audio_model.transcribe(temp_file)
                for segment in segments:
                    text += segment.text 

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text 

                result = transcription[-1]
                print(result.strip()) 

                sleep(0.25)
                result = result.strip().lower()
                        
                if "james stop" in result and play_obj is not None:
                    play_obj.stop()
                    play_obj = None

                if "hey james" in result: 

                    question_start = result.find("hey james") + len("hey james")
                    question = result[question_start:].strip() 

                    answer = ask_hugging_face(question)  # Ask GPT-3.5 Turbo
                    print("\n\n\n\n " + "User Eingabe: " + question + "\n\n\n\n " + "Open Assistent's Answer:", answer)

        except RuntimeError as error:
            print("\n\n" + "An unexpected error has occurred. Please try again." "\n\n")

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()