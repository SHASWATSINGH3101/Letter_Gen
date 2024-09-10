# Letter_Generator

<img src="https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Assets/WhatsApp%20Image%202024-09-09%20at%209.41.27%20PM.jpeg" alt="Image description">


Letter Generator is a powerful tool for creating professional letters. It uses a pre-trained model fine-tuned with formal letters to generate accurate and customized correspondence. Save time and ensure your letters are well-written and professional.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements files.

```bash
transformers
torch
streamlit
accelerate

```
Process:- 

  [Dataset Generation](https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Files/app.py)

  [Dataset Preprocessing](https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Files/data-prep.ipynb)

  [Fine Tuning](https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Files/fine-tune-peft.ipynb)

  [Modle Inference](https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Files/fine-tune-inf-peft.ipynb)

  [App File](https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Files/app.py)
  
  [Dockerfile](https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Files/Dockerfile)

HuggingFace_Space:-
[SHASWATSINGH3101/Letter_GenAI](https://huggingface.co/spaces/SHASWATSINGH3101/Letter_GenAI)

☝️☝️☝️ This is the deployment of the Fine-Tuned model on Hugginface Spaces, its very slow and it dosen't generates correct output in it for some reason.
But it generate correctly in the jupyter Notebook as you can see here:- [Model Inference](https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Files/fine-tune-inf-peft.ipynb)

--> This is the [Zero-shot Inference](https://github.com/SHASWATSINGH3101/Letter_Gen/blob/main/Files/Zero-shot%20Inference%20test) done on the base model, as you can see the generated letter has no formal letter format. 

```bibtex
@software{Letter_Generator,
  author  = {SHASWATSINGH3101},
  title   = {Letter_Generator},
  url     = {https://github.com/SHASWATSINGH3101/Letter_Generator},
  huggingface = {https://huggingface.co/SHASWATSINGH3101},
  year    = 2024,
  month   = September
}
```

## License

[Apache 2.0 license:](https://www.apache.org/licenses/LICENSE-2.0)

