# interview_ai_assistant.py

import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, simpledialog, scrolledtext, messagebox
from PyPDF2 import PdfReader
import openai
import sounddevice as sd
import queue
import whisper
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import numpy as np
import io

# Logging setup
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interview_assistant.log'),
        logging.StreamHandler()
    ]
)

# ========== GLOBAL VARIABLES ==========
audio_queue = queue.Queue()
transcript_log = []
resume_text = ""
custom_prompt = ""
selected_audio_device = None  # Store the selected audio device
is_listening = threading.Event()  # Flag to control audio listening

# ========== AUDIO CAPTURE ==========
def audio_callback(indata, frames, time_, status):
    """
    Callback function for audio input stream with enhanced logging.
    """
    if not is_listening.is_set():
        return
    
    if status:
        logging.warning(f"Audio input status: {status}")
    
    try:
        # Only add to queue if there's actual audio data and listening is active
        if indata is not None and len(indata) > 0 and is_listening.is_set():
            audio_queue.put(indata.copy())
    except Exception as e:
        logging.error(f"Error in audio callback: {e}")

def start_audio_stream(device=None):
    """
    Start audio input stream with comprehensive error handling and logging.
    
    Args:
        device (int, optional): Specific device index to use. Defaults to None.
    """
    try:
        # List available devices for debugging
        logging.info("Available Audio Devices:")
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        logging.info(f"Default Input Device: {default_input}")
        
        # Use specified device or default
        input_device = device if device is not None else default_input
        
        # Attempt to open input stream with specific parameters
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,  # Mono input
            samplerate=16000,  # Standard Whisper sample rate
            dtype='float32',
            device=input_device
        )
        
        # Start listening
        is_listening.set()
        stream.start()
        
        # Update status
        status_var.set(f"🎤 Listening on device {input_device}")
        
        return stream
    except Exception as e:
        logging.error(f"Error starting audio stream: {e}")
        status_var.set(f"❌ Audio Error: {e}")
        tk.messagebox.showerror("Audio Error", 
                                f"Could not start audio input: {e}")
        is_listening.clear()
        return None

def list_audio_devices():
    """List available audio input devices."""
    print("Available Audio Input Devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"Device {i}: {device['name']}")

def test_audio_input(duration=5, sample_rate=16000):
    """
    Test audio input and transcribe without saving files.
    
    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Audio sample rate
    """
    print(f"Testing audio input for {duration} seconds...")
    print("Please speak into your microphone...")
    
    try:
        # Record audio
        recording = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32')
        sd.wait()  # Wait until recording is finished
        
        # Flatten the recording to 1D array
        audio = recording.flatten()
        
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe directly from numpy array
        result = model.transcribe(audio)
        
        print("\n--- Transcription Results ---")
        print("Transcribed Text:", result['text'])
        print("Detected Language:", result['language'])
        
    except Exception as e:
        print(f"Error during audio test: {e}")
        import traceback
        traceback.print_exc()

# ========== TRANSCRIPTION ==========
whisper_model = whisper.load_model("base")

def transcribe_audio():
    """
    Transcribe audio with comprehensive error handling and logging.
    """
    logging.info("Audio transcription thread started.")
    audio_buffer = []
    
    try:
        while is_listening.is_set():
            try:
                # Wait for audio data with a timeout
                data = audio_queue.get(timeout=1)
                audio_buffer.append(data)
                
                # Process when buffer is sufficiently full
                if len(audio_buffer) >= 20:  # ~2 seconds of audio
                    logging.info("Transcribing audio buffer...")
                    
                    # Concatenate audio data
                    audio_np = np.concatenate(audio_buffer)
                    
                    # Ensure audio is in correct format for Whisper
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()
                    
                    # Transcribe
                    result = whisper_model.transcribe(
                        audio_np, 
                        language='en',  # Specify language to improve accuracy
                        fp16=False  # Disable FP16 to avoid potential issues
                    )
                    
                    transcript = result['text'].strip()
                    logging.info(f"Transcribed text: {transcript}")
                    
                    if transcript:
                        # Update UI from main thread
                        root.after(0, update_transcript, transcript)
                    
                    # Clear buffer
                    audio_buffer.clear()
            
            except queue.Empty:
                # No audio data received, continue waiting
                logging.debug("Waiting for audio input...")
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error in audio transcription: {e}")
                # Clear buffer on error
                audio_buffer.clear()
    
    except Exception as e:
        logging.error(f"Fatal error in audio transcription thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logging.info("Audio transcription thread stopped.")
        is_listening.clear()

def update_transcript(transcript):
    """
    Update transcript and generate response from the main thread.
    """
    try:
        transcript_log.append(("Other", transcript))
        generate_response(transcript)
    except Exception as e:
        logging.error(f"Error updating transcript: {e}")

# ========== LANGCHAIN SETUP ==========
llm = None
conversation = None


def setup_llm(api_key):
    global llm, conversation
    try:
        print(f"Attempting to set up LLM with API key: {api_key[:5]}...")
        openai.api_key = api_key
        llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o")
        
        # Verify LLM is working
        test_response = llm.invoke("Hello, can you confirm you're working?")
        print("LLM test response:", test_response)
        
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=llm, memory=memory)
        print("LLM setup completed successfully.")
    except Exception as e:
        print(f"Error setting up LLM: {e}")
        import traceback
        traceback.print_exc()
        raise

# ========== GUI ==========
root = tk.Tk()
root.title("Interview Assistant")
root.geometry("600x400")  # Larger window
root.attributes("-topmost", True)

# Create a frame for better layout
main_frame = tk.Frame(root, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# Audio Input Guidance Label
audio_guidance = tk.Label(
    main_frame, 
    text="🎤 Speak clearly into your microphone\n"
         "Ask interview questions, and I'll help you prepare answers.",
    font=("Arial", 12),
    justify=tk.CENTER,
    wraplength=500
)
audio_guidance.pack(pady=10)

# Scrolled Text for Answers
answer_box = scrolledtext.ScrolledText(
    main_frame, 
    wrap=tk.WORD, 
    height=15, 
    width=70,
    font=("Courier", 10)
)
answer_box.pack(pady=10)

# Optional: Add a status bar
status_var = tk.StringVar()
status_var.set("Ready to help with interview preparation")
status_bar = tk.Label(
    main_frame, 
    textvariable=status_var, 
    bd=1, 
    relief=tk.SUNKEN, 
    anchor=tk.W
)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

def update_display(answer):
    highlighted = highlight_star_format(answer)
    answer_box.insert(tk.END, f"\nSuggested Answer:\n{highlighted}\n")
    answer_box.see(tk.END)

# ========== PDF RESUME UPLOAD ==========
def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# ========== STAR FORMAT HIGHLIGHT ==========
def highlight_star_format(text):
    highlights = ["Situation", "Task", "Action", "Result"]
    for word in highlights:
        text = text.replace(word, f"**{word}**")
    return text

# ========== RESPONSE GENERATION ==========
def generate_response(question):
    if not llm:
        return
    prompt = f"You are helping me answer interview questions. Format answers using the STAR method when applicable (Situation, Task, Action, Result). {custom_prompt}\nResume: {resume_text}\nQuestion: {question}"
    answer = conversation.run(prompt)
    transcript_log.append(("Assistant", answer))
    update_display(answer)

# ========== TRANSCRIPT OUTPUT ==========
def export_transcript():
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for speaker, text in transcript_log:
        pdf.multi_cell(0, 10, f"{speaker}: {text}")
    pdf.output("interview_transcript.pdf")

# ========== AUDIO DEVICE SELECTION ==========
def get_input_devices():
    """
    Retrieve a list of available audio input devices.
    
    Returns:
        list: A list of dictionaries containing device information
    """
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append({
                'index': i,
                'name': device['name'],
                'default_samplerate': device.get('default_samplerate', 44100)
            })
    return input_devices

def create_device_selection_dialog(root):
    """
    Create a dialog for selecting audio input device.
    
    Args:
        root (tk.Tk): The main root window
    
    Returns:
        int: Selected device index, or None if cancelled
    """
    # Create a top-level window for device selection
    device_window = tk.Toplevel(root)
    device_window.title("Select Audio Input Device")
    device_window.geometry("500x400")
    device_window.transient(root)
    device_window.grab_set()

    # Title and instructions
    tk.Label(device_window, 
             text="Select Your Audio Input Device", 
             font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(device_window, 
             text="Choose the microphone you want to use for the interview assistant",
             wraplength=450).pack(pady=5)

    # Variable to store selected device
    selected_device = tk.IntVar()

    # Create a frame for the listbox with scrollbar
    frame = tk.Frame(device_window)
    frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    # Scrollbar
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Listbox of devices
    device_listbox = tk.Listbox(
        frame, 
        yscrollcommand=scrollbar.set, 
        font=("Courier", 10),
        selectmode=tk.SINGLE
    )
    device_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=device_listbox.yview)

    # Populate listbox with devices
    input_devices = get_input_devices()
    default_device_index = sd.default.device[0]
    
    for i, device in enumerate(input_devices):
        device_info = f"{device['index']}: {device['name']}"
        if device['index'] == default_device_index:
            device_info += " [DEFAULT]"
        device_listbox.insert(tk.END, device_info)

    # Selection variable
    selected_device_index = None

    # Submit function
    def submit_device():
        nonlocal selected_device_index
        selection = device_listbox.curselection()
        if selection:
            selected_device_index = input_devices[selection[0]]['index']
            device_window.destroy()
        else:
            tk.messagebox.showwarning("Selection", "Please select a device.")

    # Cancel function
    def cancel_selection():
        nonlocal selected_device_index
        selected_device_index = None
        device_window.destroy()

    # Buttons
    btn_frame = tk.Frame(device_window)
    btn_frame.pack(pady=10)
    
    submit_btn = tk.Button(btn_frame, text="Select Device", command=submit_device)
    submit_btn.pack(side=tk.LEFT, padx=10)
    
    cancel_btn = tk.Button(btn_frame, text="Use Default", command=cancel_selection)
    cancel_btn.pack(side=tk.LEFT, padx=10)

    # Wait for the window to be closed
    device_window.wait_window()

    return selected_device_index

# ========== SETUP DIALOGS ==========
def start_setup():
    global resume_text, custom_prompt, selected_audio_device
    try:
        logging.info("Starting setup...")
        
        # Ensure root window is fully initialized and visible
        root.update()
        root.deiconify()
        root.lift()
        root.focus_force()
        
        # Disable root window during setup to prevent interactions
        root.grab_set()
        
        # API Key Input Dialog
        api_key = None
        while not api_key:
            # Create a top-level window for API key input
            api_key_window = tk.Toplevel(root)
            api_key_window.title("API Key")
            api_key_window.geometry("300x150")
            api_key_window.transient(root)
            api_key_window.grab_set()
            
            # Label and Entry for API key
            tk.Label(api_key_window, text="Enter your OpenAI API key:").pack(pady=10)
            api_key_entry = tk.Entry(api_key_window, show='*', width=40)
            api_key_entry.pack(pady=10)
            
            # Variable to store the result
            api_key_var = tk.StringVar()
            
            # Submit function
            def submit_api_key():
                key = api_key_entry.get().strip()
                if key:
                    api_key_var.set(key)
                    api_key_window.destroy()
                else:
                    tk.messagebox.showerror("Error", "API Key cannot be empty!")
            
            # Buttons
            submit_btn = tk.Button(api_key_window, text="Submit", command=submit_api_key)
            submit_btn.pack(pady=10)
            
            # Wait for the window to be closed
            api_key_window.wait_window()
            
            # Get the API key
            api_key = api_key_var.get()
            
            if not api_key:
                # User cancelled or closed the window
                response = tk.messagebox.askretrycancel("API Key", "API Key is required to proceed.")
                if not response:
                    root.quit()
                    return
        
        logging.info("Setting up LLM...")
        setup_llm(api_key)
        logging.info("LLM setup complete.")

        # Audio Device Selection
        selected_audio_device = create_device_selection_dialog(root)
        if selected_audio_device is not None:
            logging.info(f"Selected audio device: {selected_audio_device}")
            status_var.set(f"📞 Using audio device {selected_audio_device}")
        else:
            logging.info("Using default audio device")
            status_var.set("📞 Using default audio device")

        # Custom Prompt Dialog
        custom_prompt_window = tk.Toplevel(root)
        custom_prompt_window.title("Custom Instructions")
        custom_prompt_window.geometry("400x200")
        custom_prompt_window.transient(root)
        custom_prompt_window.grab_set()
        
        tk.Label(custom_prompt_window, 
                 text="Enter specific guidelines for the assistant (optional):",
                 wraplength=350).pack(pady=10)
        
        custom_prompt_text = tk.Text(custom_prompt_window, height=5, width=50)
        custom_prompt_text.pack(pady=10)
        
        custom_prompt_var = tk.StringVar()
        
        def submit_custom_prompt():
            custom_prompt = custom_prompt_text.get("1.0", tk.END).strip()
            custom_prompt_var.set(custom_prompt)
            custom_prompt_window.destroy()
        
        submit_btn = tk.Button(custom_prompt_window, text="Submit", command=submit_custom_prompt)
        submit_btn.pack(pady=10)
        
        custom_prompt_window.wait_window()
        custom_prompt = custom_prompt_var.get()

        # Resume Upload Dialog
        filepath = filedialog.askopenfilename(
            title="Upload your resume PDF", 
            parent=root,
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if filepath:
            try:
                resume_text = extract_pdf_text(filepath)
                tk.messagebox.showinfo("Resume Uploaded", 
                                       f"Resume uploaded successfully.\n"
                                       f"Extracted {len(resume_text)} characters.",
                                       parent=root)
            except Exception as e:
                tk.messagebox.showerror("Upload Error", 
                                        f"Could not read the PDF: {str(e)}",
                                        parent=root)
                resume_text = ""
        else:
            tk.messagebox.showwarning("Resume", "No resume uploaded.", parent=root)

        # Release the grab
        root.grab_release()

        answer_box.insert(tk.END, "Setup complete. Listening for questions...\n")
        root.update()

        logging.info("Starting audio transcription thread...")
        threading.Thread(target=transcribe_audio, daemon=True).start()
        logging.info("Starting audio stream...")
        
        # Start audio stream with selected device
        stream = start_audio_stream(device=selected_audio_device)
        
        logging.info("Setup completed successfully.")
    except Exception as e:
        logging.error(f"Error in setup: {e}")
        import traceback
        traceback.print_exc()
        tk.messagebox.showerror("Setup Error", str(e), parent=root)
        root.grab_release()

# ========== EXIT HANDLER ==========
def on_closing():
    export_transcript()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# ========== START ==========
def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'list-devices':
            list_audio_devices()
            return
        elif sys.argv[1] == 'test-audio':
            test_audio_input()
            return
    
    try:
        logging.info("Initializing Interview Assistant...")
        
        # Ensure root window is fully initialized and visible
        root.update()
        root.deiconify()
        root.lift()
        root.focus_force()
        
        # Start setup in a way that doesn't block the main thread
        def delayed_setup():
            try:
                start_setup()
            except Exception as e:
                logging.error(f"Setup error: {e}")
                tk.messagebox.showerror("Setup Error", str(e))
        
        # Use after method to start setup
        root.after(100, delayed_setup)
        
        # Start the main event loop
        logging.info("Starting Tkinter main loop...")
        root.mainloop()
    except Exception as e:
        logging.error(f"Fatal error in main application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
