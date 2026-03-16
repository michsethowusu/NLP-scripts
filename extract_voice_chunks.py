import wave
import os
import sys
from pathlib import Path
import shutil

import torch
import torchaudio

INPUT_DIR = "/home/owusus/Documents/delay-data/output_chunks"
OUTPUT_DIR = "8sec_voice_only_chunks"
CHUNK_SECONDS = 8
TEMP_DIR = "temp_cleaned_audio"

# Global model loading (load once)
print("Loading Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
get_speech_timestamps, _, read_audio, _, collect_chunks = utils
print("Model loaded.\n")


def remove_non_speech(audio_path, output_path):
    """
    Remove intro/outro/transition sounds using Silero VAD.
    Keeps only speech segments.
    """
    try:
        wav = read_audio(audio_path, sampling_rate=16000)
        
        # Get speech timestamps with conservative settings
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            threshold=0.3,  # Lower = more sensitive to quiet speech
            min_speech_duration_ms=500,   # Ignore very short noises
            max_speech_duration_s=30,
            min_silence_duration_ms=300   # 300ms silence = new segment
        )
        
        if not speech_timestamps:
            print(f"    ⚠ No speech detected")
            return False
        
        # Merge close segments (remove short transition noises between speech)
        merged = []
        current = speech_timestamps[0]
        
        for next_seg in speech_timestamps[1:]:
            gap = next_seg['start'] - current['end']
            if gap < 1600:  # Less than 100ms gap (at 16kHz)
                current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
        
        # Extract speech
        speech_chunks = collect_chunks(merged, wav)
        
        # Save as WAV
        torchaudio.save(output_path, speech_chunks.unsqueeze(0), 16000)
        
        duration = len(speech_chunks) / 16000
        print(f"    ✓ Cleaned: {duration:.1f}s of speech")
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def chunk_audio(input_path, output_dir, chunk_seconds):
    """Chunk cleaned audio into fixed segments."""
    waveform, sample_rate = torchaudio.load(input_path)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    total_samples = waveform.shape[1]
    samples_per_chunk = sample_rate * chunk_seconds
    
    base_name = Path(input_path).stem.replace("cleaned_", "")
    chunk_index = 0
    samples_read = 0
    
    chunks_created = 0
    
    while samples_read < total_samples:
        end_sample = min(samples_read + samples_per_chunk, total_samples)
        chunk = waveform[:, samples_read:end_sample]
        
        # Skip short final chunks (< 2 seconds)
        if chunk.shape[1] < (2 * sample_rate):
            break
        
        out_name = f"{base_name}_chunk{chunk_index:03d}.wav"
        out_path = os.path.join(output_dir, out_name)
        
        torchaudio.save(out_path, chunk, sample_rate)
        chunks_created += 1
        
        duration = chunk.shape[1] / sample_rate
        print(f"      → {out_name} ({duration:.1f}s)")
        
        samples_read += samples_per_chunk
        chunk_index += 1
    
    return chunks_created


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    wav_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    
    if not wav_files:
        print(f"No audio files found in: {input_dir}")
        return
    
    print(f"Found {len(wav_files)} audio file(s)")
    print(f"Output: {output_dir}/")
    print(f"Chunk size: {CHUNK_SECONDS}s\n")
    
    total_chunks = 0
    
    for fname in sorted(wav_files):
        fpath = os.path.join(input_dir, fname)
        temp_path = os.path.join(TEMP_DIR, f"cleaned_{fname}")
        
        print(f"Processing: {fname}")
        
        # Step 1: Remove intros/outros/transitions
        print(f"  Cleaning non-speech...")
        success = remove_non_speech(fpath, temp_path)
        if not success:
            continue
        
        # Step 2: Chunk
        print(f"  Chunking...")
        chunks = chunk_audio(temp_path, output_dir, CHUNK_SECONDS)
        total_chunks += chunks
        print()
    
    # Cleanup
    shutil.rmtree(TEMP_DIR)
    
    print(f"Done! Created {total_chunks} chunks in {output_dir}/")


if __name__ == "__main__":
    main()
