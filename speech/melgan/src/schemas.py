def TrainInput(mel_spectrogram):
    return {
        "mel_spectrogram": mel_spectrogram,
    }

def TargetLabels(audio_waveform):
    return {"audio_waveform": audio_waveform}
