import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextStreamer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from colorama import init, Fore, Back, Style


if __name__ == "__main__":

    print(Fore.RED + "VisionLAPACK artifact:  Running artifact  ....")

    # Initialize colorama for cross-platform compatibility.
    # autoreset=True ensures the color is reset after each print statement.
    init(autoreset=True)

    #------Setup model
    
    model, tokenizer = FastVisionModel.from_pretrained(
            "/auto/projects/ChatHPC/models/cache/meta-llama/Llama-3.2-11B-Vision-Instruct",
            load_in_4bit = True,
            use_gradient_checkpointing = "unsloth",
            )
            
    model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers     = True, 
            finetune_language_layers   = True, 
            finetune_attention_modules = True,
            finetune_mlp_modules       = True,
            r = 16,           
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            random_state = 3443,
            use_rslora = False,
            loftq_config = None,
            )

    FastVisionModel.for_training(model)
    
    print(Fore.RED + "VisionLAPACK artifact:  Model set ....")


    #------Open images (matrix equations)

    imageCholeskyFactLow = Image.open("VisionLAPACK-data/Cholesky_Factorization_Lower.png")
    imageCholeskySolvLow = Image.open("VisionLAPACK-data/Cholesky_Solver_Lower.png")

    print(Fore.RED + "VisionLAPACK artifact:  Images loaded ....")

    #------Load dataset: images and prompts (LAPACK specifications)

    dataset = [ 
            {'messages': [{'role': 'user',
                'content': [{'type': 'text',
                    'text': '\nGenerate the LAPACK speficification corresponding to the matrix equation illustrated in the image\n'},
                    {'type': 'image',
                        'image': imageCholeskyFactLow}]},
                    {'role': 'assistant',
                        'content': [{'type': 'text',
                            'text':'\nDPOTRF (’L’, int n, double *a, int lda, int info)\n'
                            }]
                        }
                    ]}
            ,
            {'messages': [{'role': 'user',
                'content': [{'type': 'text',
                    'text': '\nGenerate the LAPACK speficification corresponding to the matrix equation illustrated in the image\n'},
                    {'type': 'image',
                        'image': imageCholeskySolvLow}]},
                    {'role': 'assistant',
                        'content': [{'type': 'text',
                            'text':'\nDPOTRS (’L’, int n, int nrhs, double *a, int lda, double *b, int ldb, int info)\n'
                            }]
                        }
                    ]}
]

    print(Fore.RED + "VisionLAPACK artifact:  Data set (images and LAPACK specifications) for training (fine-tuning) loaded ....")

    #------Setup training (fine-tuning)
    
    trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=dataset,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                gradient_checkpointing=True,
                warmup_steps=5,
                #max_steps=60,
                max_steps=30,
                learning_rate=2e-4,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=5,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",  # For Weights and Biases
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=4,
                max_seq_length=2048,
                #max_seq_length=4096,
                ),
            )
    
    print(Fore.RED + "VisionLAPACK artifact:  Training (fine-tuning) parameters set ....")
    
    #------Train (fine-tuning)

    print(Fore.RED + "VisionLAPACK artifact:  Training (fine-tuning) base-model with VisionLAPACK dataset ....")
    
    trainer_stats = trainer.train()
    
    #------Activate Inference

    FastVisionModel.for_inference(model)
    
    #------Testing
    
    print(Fore.RED + "VisionLAPACK artifact:  Testing Cholesky  ....")

    print(Fore.RED + "VisionLAPACK artifact:  Loading Cholesky Factorization image (matrix equation) ....")
    image = Image.open("VisionLAPACK-data/Cholesky_Factorization_Lower.png")

    print(Fore.RED + "VisionLAPACK artifact:  Generating LAPACK specification ....")
    
    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Generate the LAPACK speficification corresponding to the matrix equation illustrated in the image"}
    ]}
    ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            #max_new_tokens=128,
            max_new_tokens=1024,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )
    
    print(Fore.RED + "VisionLAPACK artifact:  Loading Cholesky Solve image (matrix equation) ....")
    image = Image.open("VisionLAPACK-data/Cholesky_Solver_Lower.png")

    print(Fore.RED + "VisionLAPACK artifact:  Generating LAPACK specification ....")
    
    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Generate the LAPACK speficification corresponding to the matrix equation illustrated in the image"}
    ]}
    ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            #max_new_tokens=128,
            max_new_tokens=1024,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )
