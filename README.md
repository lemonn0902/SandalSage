"# SandalSage" 
## Table of Contents
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## About
 This project focuses on developing a comprehensive pipeline for automatic speech recognition (ASR) for the Kannada language, particularly targeting colloquial speech used in the context of sandalwood cultivation. Using OpenAI's Whisper model, the system is fine-tuned using a custom dataset comprising audio recordings and their corresponding transcriptions. These recordings feature diverse accents, informal language styles, and background noise, reflecting real-world use cases. Additionally, we developed a question-answering module where user queries in audio form are converted to text, matched with relevant transcriptions, and the corresponding audio segments are retrieved. The speech-based question-answering module processes user queries by first converting them into text using the ASR model. Semantic search is performed using vector embeddings with tools like FAISS, alongside BM25 for traditional text retrieval. Matching segments in the audio corpus are identified by comparing embeddings of the user query and the transcriptions. LangChain framework enable efficient embedding generation, while ensemble retrievers integrate multiple techniques for improved accuracy. It includes:
- Transcribing audio files into text
- Fine Tuning
- Data Storage and Retrieval using LangChain’s FAISS and BM25Retriever

## Installation
1. Clone the repo:
   bash
   https://github.com/lemonn0902/SandalSage.git
   
2. Navigate to the project directory:
   bash
   cd SandalSage
   
3. Install dependencies:
   bash
   npm install
   

## Usage
- Run transcriptions.ipynb to generate transcriptions.csv
- create a folder named Datasets. Store the csv file and the audio (mp3) files in that folder.
(SandalSage/WhatsApp Image 2024-11-17 at 22.04.38.jpeg)
- create a copy of transcriptions.csv as meta_data.csv
- Fine tune the model by running finetuning.ipynb

bash
jupyter nbconvert --to notebook --execute finetuning.ipynb

- download the fine tuned model and unzip it in local machine.
- run the fine tuned model using streamlit run code_text_src.py
bash
streamlit run code_text_src.py



## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
Your Name - [your-email@example.com](mailto:your-email@example.com)
Project Link: [https://github.com/username/project-name](https://github.com/username/project-name)
