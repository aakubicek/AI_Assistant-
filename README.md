# Interview AI Assistant

## Overview
An AI-powered interview preparation assistant that helps users practice and improve their interview skills using OpenAI's GPT models and speech recognition.

## Features
- Real-time audio transcription
- Interview question answering with STAR method
- Custom resume integration
- Flexible audio device selection
- Customizable AI assistant instructions
- Transcript export to PDF

## Prerequisites
- Python 3.8+
- OpenAI API Key
- Working audio device (input or system audio)

## Installation
1. Clone the repository
```bash
git clone https://github.com/aakubicek/AI_Assistant-.git
cd AI_Assistant-
```
2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
Run the application:
```bash
python Navajo.py
```

### Startup Workflow
1. Enter your OpenAI API key
2. Select your preferred audio input device
3. (Optional) Provide custom AI assistant instructions
4. Upload your resume PDF
5. Start practicing interview questions!

### Command Line Options
* `python Navajo.py list-devices`: List available audio devices
* `python Navajo.py test-audio`: Test audio input/output

## Configuration
- Provide your OpenAI API key during startup
- Select your preferred audio input device
- Optionally upload a resume PDF
- Customize AI assistant behavior with optional instructions

## Technologies
- Python
- Tkinter GUI
- OpenAI GPT-4o
- Whisper Speech Recognition
- LangChain
- sounddevice for audio handling

## Troubleshooting
- Ensure you have a stable internet connection
- Check that your OpenAI API key is valid
- Verify audio device permissions

## License
MIT License

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. 