import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, os, subprocess, tempfile, time
from datetime import timedelta
import torch, numpy as np
from faster_whisper import WhisperModel
import librosa, soundfile as sf

class SubGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("SubGenerator v3.4 - CUDA 11.7 Optimized (Fixed)")
        self.root.geometry("550x420")
        self.input_file = self.output_dir = ""
        self.start_time = self.model = None
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text="SubGenerator v3.4", font=("Arial", 16, "bold")).pack(pady=(0, 10))
        ttk.Label(main_frame, text="CUDA 11.7 | GTX 1080Ti | faster-whisper 0.9.0 (Fixed)", font=("Arial", 9), foreground="blue").pack(pady=(0, 15))
        
        # Файл
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill='x', pady=5)
        ttk.Label(file_frame, text="Файл:").pack(side='left')
        self.file_label = ttk.Label(file_frame, text="Файл не выбран", foreground="gray")
        self.file_label.pack(side='left', padx=(10, 0))
        ttk.Button(file_frame, text="Выбрать", command=self.select_file).pack(side='right')
        
        # Устройство
        device_frame = ttk.Frame(main_frame)
        device_frame.pack(fill='x', pady=5)
        ttk.Label(device_frame, text="Устройство:").pack(side='left')
        self.device_var = tk.StringVar(value="auto")
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, state="readonly", width=35)
        self.device_combo.pack(side='left', padx=(10, 0))
        
        self.gpu_info_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.gpu_info_var, foreground="green", font=("Arial", 8)).pack(pady=2)
        self.setup_device_options()
        
        # Модель
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill='x', pady=5)
        ttk.Label(model_frame, text="Модель:").pack(side='left')
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=20, values=["tiny", "base", "small", "medium"])
        model_combo.pack(side='left', padx=(10, 0))
        
        # VAD настройки
        vad_frame = ttk.Frame(main_frame)
        vad_frame.pack(fill='x', pady=5)
        ttk.Label(vad_frame, text="VAD режим:").pack(side='left')
        self.vad_var = tk.StringVar(value="adaptive")
        vad_combo = ttk.Combobox(vad_frame, textvariable=self.vad_var, state="readonly", width=20, 
                                values=["adaptive", "conservative", "aggressive", "disabled"])
        vad_combo.pack(side='left', padx=(10, 0))
        
        # Папка
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill='x', pady=5)
        ttk.Label(output_frame, text="Папка:").pack(side='left')
        self.output_label = ttk.Label(output_frame, text="Рядом с файлом", foreground="gray")
        self.output_label.pack(side='left', padx=(10, 0))
        ttk.Button(output_frame, text="Выбрать папку", command=self.select_output_dir).pack(side='right')
        
        # Прогресс и статус
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100, length=450)
        self.progress_bar.pack(pady=15, fill='x')
        
        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(main_frame, textvariable=self.status_var).pack(pady=2)
        
        self.time_var = tk.StringVar(value="")
        ttk.Label(main_frame, textvariable=self.time_var, foreground="blue").pack(pady=2)
        
        self.start_button = ttk.Button(main_frame, text="Создать субтитры", command=self.start_processing)
        self.start_button.pack(pady=15)
        
        self.log_text = tk.Text(main_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill='both', expand=True, pady=(10, 0))
    
    def setup_device_options(self):
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                major, minor = torch.cuda.get_device_capability(0)
                cuda_version = torch.version.cuda
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                device_options = [f"GPU: {gpu_name} (CC {major}.{minor}, CUDA {cuda_version}, {vram_gb:.1f}GB)", "CPU (медленно, но стабильно)"]
                self.device_var.set(device_options[0])
                self.gpu_info_var.set(f"✓ {'GTX 1080Ti обнаружена - CUDA 11.7 режим' if '1080' in gpu_name and f'{major}.{minor}' == '6.1' else f'GPU обнаружена - compute capability {major}.{minor}'}")
            except Exception as e:
                device_options = ["CPU (только)", f"GPU недоступна: {str(e)}"]
                self.device_var.set("CPU (только)")
                self.gpu_info_var.set("⚠ GPU недоступна")
        else:
            device_options = ["CPU (только)"]
            self.device_var.set("CPU (только)")
            self.gpu_info_var.set("⚠ CUDA недоступна")
        
        self.device_combo['values'] = device_options
    
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_time_estimate(self, current_progress, stage_name=""):
        if self.start_time and current_progress > 5:
            elapsed = time.time() - self.start_time
            if current_progress > 0:
                total_estimated = (elapsed / current_progress) * 100
                remaining = max(0, total_estimated - elapsed)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remaining_str = str(timedelta(seconds=int(remaining)))
                self.time_var.set(f"Прошло: {elapsed_str} | Осталось: ~{remaining_str} | {stage_name}")
    
    def select_file(self):
        filename = filedialog.askopenfilename(title='Выберите аудио или видео файл',
            filetypes=[('Аудио и видео', '*.wav *.mp4 *.avi *.mkv *.mov *.mp3 *.flac *.m4a *.webm')])
        if filename:
            self.input_file = filename
            self.file_label.config(text=os.path.basename(filename), foreground="black")
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(title='Выберите папку для сохранения')
        if directory:
            self.output_dir = directory
            self.output_label.config(text=directory, foreground="black")
    
    def start_processing(self):
        if not self.input_file:
            messagebox.showerror("Ошибка", "Выберите файл!")
            return
        self.start_button.config(state="disabled")
        self.start_time = time.time()
        threading.Thread(target=self.process_file, daemon=True).start()
    
    def extract_audio_optimized(self, video_path, output_path):
        try:
            self.log_message("Извлечение аудио (16kHz, mono)...")
            # Улучшенная команда FFmpeg с фильтрацией шума и нормализацией
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-vn',  # Только аудио
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',  # 16kHz
                '-ac', '1',  # mono
                '-af', 'highpass=f=80,lowpass=f=8000,volume=1.2',  # Фильтры для очистки
                '-y', output_path
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.log_message("✓ Аудио обработано с фильтрацией")
            return True
        except subprocess.CalledProcessError as e:
            self.log_message(f"Ошибка FFmpeg: {e.stderr}")
            return False
        except Exception as e:
            self.log_message(f"Ошибка извлечения аудио: {e}")
            return False
    
    def get_device_settings(self):
        device_text = self.device_var.get()
        return ("cpu", "int8") if "CPU" in device_text or not torch.cuda.is_available() else ("cuda", "int8_float32")
    
    def get_vad_parameters(self):
        """Возвращает параметры VAD в зависимости от выбранного режима"""
        vad_mode = self.vad_var.get()
        
        if vad_mode == "disabled":
            return None
        elif vad_mode == "conservative":
            return dict(
                threshold=0.3,
                min_speech_duration_ms=100,
                max_speech_duration_s=30,
                min_silence_duration_ms=200,
                window_size_samples=512,
                speech_pad_ms=200
            )
        elif vad_mode == "aggressive":
            return dict(
                threshold=0.7,
                min_speech_duration_ms=50,
                max_speech_duration_s=15,
                min_silence_duration_ms=100,
                window_size_samples=256,
                speech_pad_ms=100
            )
        else:  # adaptive (по умолчанию)
            return dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_s=20,
                min_silence_duration_ms=300,
                window_size_samples=1024,
                speech_pad_ms=300
            )
    
    def load_whisper_model(self):
        try:
            device, compute_type = self.get_device_settings()
            model_size = self.model_var.get()
            self.log_message(f"Загрузка faster-whisper {model_size} модели...")
            self.log_message(f"Устройство: {device}, тип: {compute_type}")
            
            model_params = {
                'model_size_or_path': model_size,
                'device': device,
                'compute_type': compute_type,
                'num_workers': 1,
                'download_root': None,
                'local_files_only': False
            }
            
            if device == "cuda":
                model_params['cpu_threads'] = 4
            else:
                model_params['cpu_threads'] = 8
            
            self.model = WhisperModel(**model_params)
            self.log_message("✓ Модель загружена успешно")
            return True
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки модели: {str(e)}")
            return False
    
    def clean_transcription_artifacts(self, word_timings):
        """Очищает артефакты транскрипции типа 'АААА' и 'эээээ'"""
        cleaned_words = []
        
        for word_data in word_timings:
            word = word_data['word'].strip()
            
            # Пропускаем явные артефакты
            if self.is_transcription_artifact(word):
                self.log_message(f"🧹 Удален артефакт: '{word}' ({word_data['start']:.1f}s)")
                continue
            
            # Фильтруем слишком низкую уверенность
            if word_data.get('probability', 1.0) < 0.3:
                self.log_message(f"🧹 Удалено слово с низкой уверенностью: '{word}' (p={word_data.get('probability', 0):.2f})")
                continue
            
            cleaned_words.append(word_data)
        
        return cleaned_words
    
    def is_transcription_artifact(self, word):
        """Определяет, является ли слово артефактом транскрипции"""
        word_lower = word.lower().strip()
        
        # Проверяем на повторяющиеся символы
        if len(word_lower) > 3:
            unique_chars = set(word_lower)
            if len(unique_chars) <= 2:  # Слово состоит из 1-2 уникальных символов
                return True
        
        # Проверяем длинные последовательности одного символа
        for char in ['а', 'э', 'о', 'у', 'и', 'ы', 'е', 'm', 'n', 'h']:
            if word_lower.count(char) > 4:
                return True
        
        # Проверяем паттерны артефактов
        artifacts = [
            'ааааа', 'эээээ', 'ооооо', 'ууууу', 'ммммм',
            'нннн', 'хммм', 'эмм', 'амм', 'угу', 'мгм'
        ]
        
        if word_lower in artifacts or any(artifact in word_lower for artifact in artifacts):
            return True
        
        return False
    
    def transcribe_with_word_timestamps(self, audio_path):
        try:
            if not self.model and not self.load_whisper_model():
                return []
            
            self.log_message("Распознавание речи с улучшенными параметрами...")
            self.progress_var.set(35)
            self.update_time_estimate(35, "Распознавание речи")
            
            # Получаем параметры VAD
            vad_params = self.get_vad_parameters()
            vad_enabled = vad_params is not None
            
            self.log_message(f"VAD режим: {self.vad_var.get()} {'(включен)' if vad_enabled else '(отключен)'}")
            
            # Улучшенные параметры транскрипции
            transcribe_params = {
                'audio': audio_path,
                'language': 'ru',
                'word_timestamps': True,
                'beam_size': 5,
                'best_of': 5,  # Увеличиваем для лучшего качества
                'temperature': [0.0, 0.2, 0.4],  # Многоступенчатая температура
                'condition_on_previous_text': True,  # Включаем контекст
                'compression_ratio_threshold': 2.4,
                'no_speech_threshold': 0.6,
                'initial_prompt': "Это русская речь. Избегайте повторяющихся звуков и артефактов.",
                'vad_filter': vad_enabled
            }
            
            if vad_enabled:
                transcribe_params['vad_parameters'] = vad_params
            
            segments, info = self.model.transcribe(**transcribe_params)
            
            self.progress_var.set(70)
            self.update_time_estimate(70, "Обработка результатов")
            
            word_timings = []
            total_segments = 0
            processed_segments = 0
            
            for segment in segments:
                total_segments += 1
                
                # Проверяем качество сегмента
                if hasattr(segment, 'avg_logprob') and segment.avg_logprob < -1.5:
                    self.log_message(f"⚠ Пропущен сегмент с низким качеством: {segment.start:.1f}s-{segment.end:.1f}s")
                    continue
                
                processed_segments += 1
                
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        if hasattr(word, 'word') and hasattr(word, 'start') and hasattr(word, 'end'):
                            word_timings.append({
                                "word": word.word.strip(),
                                "start": word.start,
                                "end": word.end,
                                "probability": getattr(word, 'probability', 0.8)
                            })
                else:
                    # Fallback для сегментов без word timestamps
                    text = segment.text.strip()
                    if text and len(text) > 0:
                        words = text.split()
                        if words:
                            duration = segment.end - segment.start
                            word_duration = duration / len(words)
                            for i, word in enumerate(words):
                                word_start = segment.start + (i * word_duration)
                                word_timings.append({
                                    "word": word,
                                    "start": word_start,
                                    "end": word_start + word_duration,
                                    "probability": 0.7
                                })
            
            # Очищаем артефакты
            original_count = len(word_timings)
            word_timings = self.clean_transcription_artifacts(word_timings)
            cleaned_count = len(word_timings)
            
            self.log_message(f"✓ Распознано сегментов: {total_segments} (обработано: {processed_segments})")
            self.log_message(f"✓ Извлечено слов: {original_count} (после очистки: {cleaned_count})")
            self.log_message(f"✓ Удалено артефактов: {original_count - cleaned_count}")
            self.log_message(f"✓ Язык: {info.language} (уверенность: {info.language_probability:.2f})")
            
            return word_timings
            
        except Exception as e:
            self.log_message(f"❌ Ошибка транскрипции: {str(e)}")
            return []
    
    def format_time(self, seconds):
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def create_srt_file(self, word_timings, output_path):
        try:
            self.log_message("Создание SRT файла...")
            
            # Сортируем по времени на всякий случай
            word_timings.sort(key=lambda x: x['start'])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, word_data in enumerate(word_timings, 1):
                    start = self.format_time(word_data['start'])
                    end = self.format_time(word_data['end'])
                    word = word_data['word'].strip()
                    
                    # Дополнительная проверка на артефакты при записи
                    if not self.is_transcription_artifact(word) and len(word) > 0:
                        f.write(f"{i}\n{start} --> {end}\n{word}\n\n")
            
            return True
        except Exception as e:
            self.log_message(f"❌ Ошибка создания SRT: {str(e)}")
            return False
    
    def process_file(self):
        try:
            self.status_var.set("Обработка...")
            self.progress_var.set(0)
            
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            output_dir = self.output_dir if self.output_dir else os.path.dirname(self.input_file)
            output_path = os.path.join(output_dir, f"{base_name}_fixed.srt")
            
            file_ext = os.path.splitext(self.input_file)[1].lower()
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
            cleanup_files = []
            
            if file_ext in video_extensions or (file_ext in audio_extensions and file_ext != '.wav'):
                temp_audio = os.path.join(output_dir, f"temp_{base_name}_processed.wav")
                if not self.extract_audio_optimized(self.input_file, temp_audio):
                    raise Exception("Не удалось извлечь/конвертировать аудио")
                audio_path = temp_audio
                cleanup_files = [temp_audio]
            elif file_ext in audio_extensions:
                audio_path = self.input_file
            else:
                raise Exception(f"Неподдерживаемый формат файла: {file_ext}")
            
            self.progress_var.set(15)
            self.update_time_estimate(15, "Подготовка аудио завершена")
            
            if not self.model and not self.load_whisper_model():
                raise Exception("Не удалось загрузить модель Whisper")
            
            self.progress_var.set(25)
            self.update_time_estimate(25, "Модель готова")
            
            word_timings = self.transcribe_with_word_timestamps(audio_path)
            if not word_timings:
                raise Exception("Не удалось распознать речь или извлечь слова")
            
            self.progress_var.set(85)
            self.update_time_estimate(85, "Создание субтитров")
            
            if not self.create_srt_file(word_timings, output_path):
                raise Exception("Не удалось создать SRT файл")
            
            # Очистка временных файлов
            for temp_file in cleanup_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                        self.log_message(f"🗑️ Удален временный файл: {os.path.basename(temp_file)}")
                    except:
                        pass
            
            self.progress_var.set(100)
            self.status_var.set("Готово!")
            
            total_time = time.time() - self.start_time
            total_time_str = str(timedelta(seconds=int(total_time)))
            
            self.time_var.set(f"Завершено за: {total_time_str}")
            self.log_message(f"✅ SRT создан: {output_path}")
            self.log_message(f"📊 Всего слов: {len(word_timings)}")
            self.log_message(f"⏱️ Время обработки: {total_time_str}")
            
            probabilities = [w.get('probability', 0) for w in word_timings if w.get('probability')]
            if probabilities:
                avg_confidence = sum(probabilities) / len(probabilities)
                self.log_message(f"📈 Средняя уверенность: {avg_confidence:.2f}")
            
            messagebox.showinfo("Успех", f"SRT файл создан!\n{output_path}\n\nВсего слов: {len(word_timings)}\nВремя: {total_time_str}\nМодель: {self.model_var.get()}\nVAD: {self.vad_var.get()}")
            
        except Exception as e:
            self.status_var.set("Ошибка")
            self.time_var.set("")
            self.log_message(f"❌ ОШИБКА: {str(e)}")
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.start_button.config(state="normal")

def main():
    print("=== SubGenerator v3.4 - System Check (Fixed Version) ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        major, minor = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {major}.{minor}")
    print("========================================================")
    
    root = tk.Tk()
    app = SubGenerator(root)
    root.mainloop()

if __name__ == "__main__":
    main()