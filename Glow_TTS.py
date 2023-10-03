import os

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from trainer import Trainer, TrainerArgs

#Loading your dataset
output_path = "tts_train_dir"
if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "LJSpeech-1.1/")
)

#Train a new model
# GlowTTSConfig: all model related values for training, validating and testing.

config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    save_step=1000,
)
#Next we will initialize the audio processor which is used for feature extraction and audio I/O.
ap = AudioProcessor.init_from_config(config)

#Next we will initialize the tokenizer which is used to convert text to sequences of token IDs. If characters are not defined in the config, default characters are passed to the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

#Next we will load data samples. Each sample is a list of [text, audio_file_path, speaker_name]. You can define your custom sample loader returning the list of samples.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)
#Now we're ready to initialize the model.
#Models take a config object and a speaker manager as input. Config defines the details of the model like the number of layers, the size of the embedding, etc. Speaker manager is used by multi-speaker models.


model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

#Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training, distributed training, etc.

trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

#AND... 3,2,1... START TRAINING 🚀🚀🚀
trainer.fit()

