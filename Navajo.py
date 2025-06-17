# Standard library imports
import os
import sys
import time
import io
import queue
import logging
import threading
import traceback
from collections import defaultdict

# Third-party library imports
import numpy as np
import openai
import whisper
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, simpledialog, scrolledtext, messagebox
from PyPDF2 import PdfReader
from pydub import AudioSegment
from fpdf import FPDF

# Machine Learning and AI imports
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interview_assistant.log'),
        logging.StreamHandler()
    ]
)

# ========== GLOBAL VARIABLES ==========
# Global flag to track interview status
print("Setting up global variables...")
interview_completed = False
audio_queue = queue.Queue()
transcript_log = []
resume_text = ""
custom_prompt = ""
selected_audio_device = None  # Store the selected audio device
is_listening = threading.Event()  # Flag to control audio listening
is_paused = False

# ========== AUDIO CAPTURE ==========
def preprocess_audio(audio_data):
    """Preprocess audio data to enhance quality."""
    # Convert to AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=16000,
        sample_width=2,  # Assuming 16-bit audio
        channels=1
    )
    
    # Normalize audio
    normalized_audio = audio_segment.apply_gain(-audio_segment.dBFS)
    
    return normalized_audio

def audio_callback(outdata, frames, time_, status):
    """Callback function for audio output stream."""
    if status:
        print(status)
    if is_listening.is_set() and not is_paused:
        # Capture audio data
        audio_data = np.frombuffer(outdata, dtype=np.float32)
        processed_audio = preprocess_audio(audio_data)
        audio_queue.put(processed_audio)

def list_audio_devices():
    """List available audio output devices."""
    print("Available Audio Output Devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            print(f"Device {i}: {device['name']}")

def test_audio_input(duration=5, sample_rate=16000):
    """
    Test system audio output and transcribe.
    
    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Audio sample rate
    """
    print(f"Testing system audio output for {duration} seconds...")
    print("Please play audio on your system...")
    
    try:
        # Record system audio
        recording = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32',
                           device=sd.default.device[1])  # Use default output device
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
        print(f"Error during system audio test: {e}")
        import traceback
        traceback.print_exc()

# ========== TRANSCRIPTION ==========
whisper_model = whisper.load_model("base")

def transcribe_audio():
    """Transcribe audio with comprehensive error handling and logging."""
    logging.info("Audio transcription thread started.")
    audio_buffer = []
    
    try:
        while is_listening.is_set():
            if is_paused:
                time.sleep(0.5)  # Wait if paused
                continue
            
            try:
                # Wait for audio data with a timeout
                data = audio_queue.get(timeout=1)
                audio_buffer.append(data)
                
                # Process when buffer is sufficiently full
                if len(audio_buffer) >= 20:  # ~2 seconds of audio
                    logging.info("Transcribing audio buffer...")
                    
                    # Concatenate audio data
                    audio_np = np.concatenate(audio_buffer)
                    
                    # Transcribe using Whisper
                    result = whisper_model.transcribe(
                        audio_np,
                        language='en',
                        task='transcribe',
                        fp16=False
                    )
                    
                    transcript = result['text'].strip()
                    logging.info(f"Transcribed text: {transcript}")
                    
                    if transcript:
                        # Update UI from main thread
                        root.after(0, update_transcript, transcript)
                    
                    # Clear buffer
                    audio_buffer.clear()
            
            except queue.Empty:
                logging.debug("Waiting for audio input...")
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error in audio transcription: {e}")
                audio_buffer.clear()
    
    except Exception as e:
        logging.error(f"Fatal error in audio transcription thread: {e}")
    finally:
        logging.info("Audio transcription thread stopped.")
        is_listening.clear()

def update_transcript(transcript):
    """
    Update transcript and generate response from the main thread.
    Modified to enable Interview Complete button when responses start.
    """
    try:
        # Only enable the button after first transcript entry
        if interview_complete_btn['state'] == tk.DISABLED:
            interview_complete_btn.config(state=tk.NORMAL)
        
        transcript_log.append(("Other", transcript))
        generate_response(transcript)
    except Exception as e:
        logging.error(f"Error updating transcript: {e}")
        import traceback
        traceback.print_exc()

# ========== LANGCHAIN SETUP ==========
llm = None
conversation = None


def setup_llm(api_key):
    global llm, conversation
    try:
        print(f"Attempting to set up LLM with API key: {api_key[:5]}...")
        openai.api_key = api_key
        
        # Use GPT-4o mini model
        llm = ChatOpenAI(
            openai_api_key=api_key, 
            model_name="gpt-4o-mini",  # Specify the GPT-4o mini model
            temperature=0.7,  # Add some creativity for interview feedback
            max_tokens=300   # Limit response length for concise feedback
        )
        
        # Verify LLM is working
        test_response = llm.invoke("Hello, can you confirm you're working?")
        print("LLM test response:", test_response)
        
        # Setup conversation chain with memory
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm, 
            memory=memory,
            verbose=True  # Enable verbose mode for more detailed logging
        )
        print("LLM setup completed successfully.")
    except Exception as e:
        print(f"Error setting up LLM: {e}")
        import traceback
        traceback.print_exc()
        raise

# ========== GUI ==========
print("Initializing Tkinter root window...")
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
    text="🎤 \n"
         "Have the interviewer ask questions, and I'll help you prepare answers.",
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
    """
    Generate a summary of the interview transcript with an AI-powered performance review.
    Allows user to choose where to save the PDF.
    """
    from fpdf import FPDF
    from collections import defaultdict
    import os
    import sys
    
    # Ensure we have a conversation chain
    if not conversation:
        tk.messagebox.showerror("Error", "LLM not initialized for performance review")
        return
    
    # Only generate summary if interview was actually completed
    if not interview_completed or len(transcript_log) < 2:
        tk.messagebox.showwarning("Warning", "No interview data to summarize.")
        return
    
    # Group transcript by speaker
    speaker_summaries = defaultdict(list)
    for speaker, text in transcript_log:
        speaker_summaries[speaker].append(text)
    
    # Generate performance review using the LLM
    try:
        performance_review_prompt = (
            "Provide a professional performance review based on the interview responses. "
            "Include a numerical rating out of 10, highlighting strengths and areas for improvement. "
            "Be constructive and specific. The review should be concise but informative. "
            "Responses were: " + " ".join(speaker_summaries.get("Other", []))
        )
        performance_review = conversation.run(performance_review_prompt)
    except Exception as e:
        logging.error(f"Error generating performance review: {e}")
        performance_review = "Unable to generate performance review due to an error."
    
    # Open file dialog to choose save location
    pdf_filename = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        title="Save Interview Performance Summary"
    )
    
    # If user cancels save dialog
    if not pdf_filename:
        tk.messagebox.showinfo("Save Cancelled", "PDF export was cancelled.")
        return
    
    # Create PDF with summary and review
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Interview Performance Summary", ln=True)
    pdf.ln(10)
    
    # Performance Review Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Performance Review", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, performance_review)
    pdf.ln(10)
    
    # Transcript Summary Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Transcript Summary", ln=True)
    pdf.set_font("Arial", size=12)
    
    # Summarize each speaker's contributions
    for speaker, texts in speaker_summaries.items():
        # Combine texts and truncate if too long
        combined_text = " ".join(texts)
        
        # Basic summarization: first 200 characters and ellipsis if longer
        summary = combined_text[:200] + (f"... (truncated)" if len(combined_text) > 200 else "")
        
        # Add speaker summary
        pdf.cell(0, 10, f"{speaker} Summary:", ln=True, style='B')
        pdf.multi_cell(0, 10, summary)
        pdf.ln(5)  # Add some spacing between summaries
    
    # Add timestamp
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    # Output the PDF to chosen location
    pdf.output(pdf_filename)
    
    # Show a message that summary is ready
    tk.messagebox.showinfo("Performance Summary", 
                            f"Interview performance summary has been saved to {pdf_filename}")
    
    # Reset interview status
    global interview_completed
    interview_completed = False
    
    # Reset UI
    interview_complete_btn.config(state=tk.NORMAL)
    status_var.set("Ready for next interview")

# ========== AUDIO DEVICE SELECTION ==========
def get_input_devices():
    """
    Retrieve a list of available audio output devices.
    
    Returns:
        list: A list of dictionaries containing device information
    """
    devices = sd.query_devices()
    output_devices = []
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            output_devices.append({
                'index': i,
                'name': device['name'],
                'default_samplerate': device.get('default_samplerate', 44100)
            })
    return output_devices

def create_device_selection_dialog(root):
    """
    Create a dialog for selecting audio output device.
    
    Args:
        root (tk.Tk): The main root window
    
    Returns:
        int: Selected device index, or None if cancelled
    """
    # Create a top-level window for device selection
    device_window = tk.Toplevel(root)
    device_window.title("Select System Audio Output Device")
    device_window.geometry("500x400")
    device_window.transient(root)
    device_window.grab_set()

    # Title and instructions
    tk.Label(device_window, 
             text="Select Your System Audio Output Device", 
             font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(device_window, 
             text="Choose the audio output device you want to listen to",
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

    # Populate listbox with output devices
    output_devices = [
        device for device in sd.query_devices() 
        if device['max_output_channels'] > 0
    ]
    default_device_index = sd.default.device[1]
    
    for i, device in enumerate(output_devices):
        device_info = f"{i}: {device['name']}"
        if i == default_device_index:
            device_info += " [DEFAULT]"
        device_listbox.insert(tk.END, device_info)

    # Selection variable
    selected_device_index = None

    # Submit function
    def submit_device():
        nonlocal selected_device_index
        selection = device_listbox.curselection()
        if selection:
            selected_device_index = selection[0]
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
    
    print("Starting Interview Assistant...")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'list-devices':
            list_audio_devices()
            return
        elif sys.argv[1] == 'test-audio':
            test_audio_input()
            return
    
    try:
        print("Initializing Interview Assistant...")
        
        # Ensure root window is fully initialized and visible
        root.update()
        root.deiconify()
        root.lift()
        root.focus_force()
        
        # Start setup in a way that doesn't block the main thread
        def delayed_setup():
            try:
                print("Running delayed setup...")
                start_setup()
            except Exception as e:
                print(f"Setup error: {e}")
                import traceback
                traceback.print_exc()
                tk.messagebox.showerror("Setup Error", str(e))
        
        # Use after method to start setup
        root.after(100, delayed_setup)
        
        # Start the main event loop
        print("Starting Tkinter main loop...")
        root.mainloop()
    except Exception as e:
        print(f"Fatal error in main application: {e}")
        import traceback
        traceback.print_exc()

def start_audio_stream(device=None):
    """Start audio stream with comprehensive error handling and logging."""
    try:
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=16000,
            dtype='float32'
        )
        
        is_listening.set()
        stream.start()
        
        return stream
    except Exception as e:
        logging.error(f"Error starting audio stream: {e}")
        is_listening.clear()
        return None

# Create a frame for control buttons
control_frame = tk.Frame(main_frame)
control_frame.pack(pady=10)

# Pause Interview Button
pause_btn = tk.Button(
    control_frame, 
    text="⏸️ Pause Interview", 
    command=lambda: toggle_pause(),
    bg='lightyellow',
    font=("Arial", 12, "bold")
)
pause_btn.pack(side=tk.LEFT, padx=5)

# Interview Complete Button
interview_complete_btn = tk.Button(
    control_frame, 
    text="🏁 Interview Complete", 
    command=lambda: end_interview(),
    bg='lightgreen',
    font=("Arial", 12, "bold"),
    state=tk.NORMAL  # Initially enabled
)
interview_complete_btn.pack(side=tk.LEFT, padx=5)

def end_interview():
    """
    Handle the end of the interview process.
    Stops listening, enables summary generation, and triggers export.
    """
    global is_listening, interview_completed
    
    # Stop audio listening
    is_listening.clear()
    
    # Mark interview as completed
    interview_completed = True
    
    # Disable the complete button after it's pressed
    interview_complete_btn.config(state=tk.DISABLED)
    
    # Update status
    status_var.set("🏁 Interview Completed. Generating Summary...")
    
    # Generate and export transcript
    export_transcript()
    
    # Show completion message
    tk.messagebox.showinfo("Interview Complete", 
                            "Interview summary has been generated.")

def toggle_pause():
    """
    Toggle between paused and unpaused states during the interview.
    """
    global is_paused, is_listening
    
    if not is_paused:
        # Pause the interview
        is_paused = True
        is_listening.clear()
        pause_btn.config(
            text="▶️ Unpause Interview", 
            bg='lightgreen'
        )
        status_var.set("⏸️ Interview Paused")
        
        # Optional: Add a visual indicator of pause
        audio_guidance.config(
            text="🛑 Interview Paused\n"
                 "Click 'Unpause' to continue",
            fg='red'
        )
    else:
        # Unpause the interview
        is_paused = False
        is_listening.set()
        pause_btn.config(
            text="⏸️ Pause Interview", 
            bg='lightyellow'
        )
        status_var.set("🎤 Interview Resumed")
        
        # Restore original guidance text
        audio_guidance.config(
            text="🎤 \n"
                 "Have the interviewer ask questions, and I'll help you prepare answers.",
            fg='black'
        )
        
        # Restart audio stream if needed
        try:
            start_audio_stream(device=selected_audio_device)
        except Exception as e:
            logging.error(f"Error restarting audio stream: {e}")
            tk.messagebox.showerror("Audio Error", 
                                    f"Could not resume audio stream: {e}")

if __name__ == "__main__":
    main()
