import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.restore_from(
    restore_path="models/Parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo",
    map_location="cuda"
)

output = asr_model.transcribe(
    ["recording.wav"]
)

print(output[0].text)