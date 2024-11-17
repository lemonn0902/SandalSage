## SandalSage 
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
1.⁠ ⁠Clone the repo:
   '''
   ⁠bash
   https://github.com/lemonn0902/SandalSage.git
   '''
   
    ⁠
2.⁠ ⁠Navigate to the project directory:
   ⁠bash
   cd SandalSage
   
    ⁠
3.⁠ ⁠Install dependencies:
   ⁠bash
   npm install

   

## Usage
- Run transcriptions.ipynb to generate transcriptions.csv
- create a folder named Datasets. Store the csv file and the audio (mp3) files in that folder.
  pic 1
- create a copy of transcriptions.csv as meta_data.csv
- Fine tune the model by running finetuning.ipynb


bash
jupyter nbconvert --to notebook --execute finetuning.ipynb
 ⁠
-  ⁠download the fine tuned model and unzip it in local machine.
-  ⁠run the fine tuned model using streamlit run code_text_src.py and create a file structure like the picture below using:

- ![image](https://github.com/user-attachments/assets/719a389d-229a-4c2f-a7e2-10acf2e0f856)

⁠ bash
streamlit run code_text_src.py


- Project Structure:
- ![image](https://github.com/user-attachments/assets/d350ddb0-d987-4a00-a2d4-b082ccee3428)




## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
Team members mail-ids - [Eshithachowdary](mailto:eshithachowdary.cs23@rvce.edu.in)
      [Shreya](mailto:shreyas.cy23@rvce.edu.in)
      [D Rakshitha](mailto:drakshitha.cs23@rvce.edu.in)
Project Link: [https://github.com/suzy2521alien/SandalSage](https://github.com/suzy2521alien/SandalSage)
