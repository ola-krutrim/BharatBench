# BharatBench

## Overview

**BharatBench** is a comprehensive vision-language benchmark framework designed to evaluate models based on India's rich linguistic diversity. It focuses on testing the capabilities of vision-language models across multiple Indian languages to ensure inclusivity and performance consistency in multilingual contexts.

## Features

- Supports a wide range of Indian languages, ensuring models can handle the complexity and nuances of each.
- Provides robust benchmarks for evaluating vision-language models in real-world, culturally relevant scenarios.
- Standardized evaluation metrics to compare models across languages.

## Supported Languages

BharatBench is designed to test vision-language models across the following 12 languages:

- Hindi
- Telugu
- Marathi
- Gujarati
- Kannada
- Malayalam
- Tamil
- Sanskrit
- Odia
- Assamese
- Bengali
- English

## Usage

1. **Prepare your model**: Ensure your vision-language model supports multimodal inputs (image + text) and can provide outputs in multiple languages.
2. **Run evaluations**: Use the benchmarks provided in BharatBench to evaluate your model's performance across the supported Indian languages.
3. **Compare results**: Analyze the results using the standardized metrics and compare performance across different languages.

## Repository Structure
1. **Indic-MMVet-Evaluator**: switch to MMVet-Evaluator and refer to [ReadMe.md](./Indic-MMVet-Evaluator/ReadMe.md) to know how to run MMVet evaluations for Indian languages.
2. **Indic-LLaVABench-Evaluator**: switch to LLaVA-Bench-Evaluator and refer to [ReadMe.md](./Indic-LLaVABench-Evaluator/ReadMe.md) to know how to run LLaVA-Bench evaluations for Indian languages.
3. **Indic-POPE-Evaluator**: switch to POPE-Evaluator and refer to [ReadMe.md](./Indic-POPE-Evaluator/ReadMe.md) to know how to run POPE evaluations for Indian languages.

## Images for the above mentioned benchmarks
We assess the Bharat linguistic capabilities of VLMs using evaluators like **Indic-MMVet**, **Indic-LLaVABench**, and **Indic-POPE**, with standard images from MMVet, POPE, and LLaVABench.


1. **Indic-MMVet-Evaluator**: https://huggingface.co/datasets/krutrim-ai-labs/IndicMMVet
2. **Indic-LLaVABench-Evaluator**: https://huggingface.co/datasets/krutrim-ai-labs/IndicLLaVABench
3. **Indic-POPE-Evaluator**: https://huggingface.co/datasets/krutrim-ai-labs/IndicPope

## Get Started

Clone the repository and follow the instructions provided in the respective scripts to start evaluating your vision-language models.

```bash
git clone https://github.com/your-username/BharatBench.git
cd BharatBench
```

## License
This code repository and the model weights are licensed under the [Krutrim Community License Agreement Version 1.0](LICENSE)


## Contact
Contributions are welcome! If you have any improvements or suggestions, feel free to submit a pull request on GitHub.

## Citation
If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:

```bash
@article{
  khan2025chitrarth,
  title={Chitrarth: Bridging Vision and Language for a Billion People},
  author={Shaharukh Khan, Ayush Tarun, Abhinav Ravi, Ali Faraz, Akshat Patidar, Praveen Kumar Pokala, Anagha Bhangare, Raja Kolla, Chandra Khatri, Shubham Agarwal},
  journal={arXiv preprint arXiv:2502.15392},
  year={2025}
}

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}

@article{gala2023indictrans2,
  title={Indictrans2: Towards high-quality and accessible machine translation models for all 22 scheduled indian languages},
  author={Gala, Jay and Chitale, Pranjal A and AK, Raghavan and Gumma, Varun and Doddapaneni, Sumanth and Kumar, Aswanth and Nawale, Janki and Sujatha, Anupama and Puduppully, Ratish and Raghavan, Vivek and others},
  journal={arXiv preprint arXiv:2305.16307},
  year={2023}
}

```

## Acknowledgement

BharatBench is built with reference to the code of the following projects: [LLaVA-1.5](https://github.com/haotian-liu/LLaVA/tree/main/llava/eval). Thanks for their awesome work!
