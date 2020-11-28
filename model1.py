import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

from torchsummary import summary
from constants import SAMPLE_RATE, N_MELS, N_FFT, F_MAX, F_MIN, HOP_SIZE


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspectrogram = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_SIZE, f_min=F_MIN, f_max=F_MAX, n_mels=N_MELS, normalized=False)
    
    def forward(self, audio):
        batch_size = audio.shape[0]
        
        # alignment correction to match with pianoroll
        # pretty_midi.get_piano_roll use ceil, but torchaudio.transforms.melspectrogram uses
        # round when they convert the input into frames.
        padded_audio = nn.functional.pad(audio, (N_FFT // 2, 0), 'constant')
        mel = self.melspectrogram(audio)[:, :, 1:]
        mel = mel.transpose(-1, -2)
        mel = th.log(th.clamp(mel, min=1e-9))
        return mel



class LSTMs(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden_size=1024, output_size=88, num_layers=2, bidirectional=True):
        super().__init__()
        self.n_mels, self.hidden_size, self.output_size = n_mels, hidden_size, output_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=self.n_mels,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.output_fc = nn.Linear(self.hidden_size * self.num_directions, output_size)
#        self.sig = nn.Sigmoid()

    def forward(self, x):
        self.batch_size = x.size(0)
        h0 = self.init_hidden()
        c0 = self.init_hidden()

        output, hidden = self.lstm(x, (h0, c0))
        # output.shape(batch_size, seq_len, hidden_size * direction)
        output = self.output_fc(output[:, :, -1])
#        output = self.sig(output)
        return output

    def init_hidden(self):
        return th.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)


class Transcriber(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc = nn.Linear(fc_unit, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_fc = nn.Linear(fc_unit, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio) 

        x = self.frame_conv_stack(mel)  # (B x T x C)
        frame_out = self.frame_fc(x)

        x = self.onset_conv_stack(mel)  # (B x T x C)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out


class Transcriber_RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()

        self.frame_LSTM = LSTMs()  # Parameters for LSTM are defined as default.
#        self.frame_fc = nn.Linear(256, 200*88)
        self.frame_fc = self.frame_LSTM

        self.onset_LSTM = LSTMs()
#        self.onset_fc = nn.Linear(256, 200*88)
        self.onset_fc = self.onset_LSTM

    def forward(self, audio):
        mel = self.melspectrogram(audio)

        x = self.frame_LSTM(mel)
        frame_out = self.frame_fc(x)
        
        x = self.onset_LSTM(mel)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out
