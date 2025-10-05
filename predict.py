from cog import BasePredictor, Input, Path
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

class MultiModalCallEvalModel(nn.Module):
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", 
                 bert_model_name="bert-base-uncased", dropout=0.3):
        super().__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        self.audio_projection = nn.Linear(self.wav2vec2.config.hidden_size, 256)
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.text_projection = nn.Linear(self.bert.config.hidden_size, 256)
        
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.phase_classifier = nn.Linear(128, 3)
        self.filler_classifier = nn.Linear(128, 1)
        self.quality_regressor = nn.Linear(128, 1)
        self.enthusiasm_classifier = nn.Linear(128, 1)
        self.politeness_classifier = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, audio_input_values, audio_attention_mask, 
                text_input_ids, text_attention_mask):
        
        audio_outputs = self.wav2vec2(
            input_values=audio_input_values,
            attention_mask=audio_attention_mask
        )
        
        audio_embeddings = audio_outputs.last_hidden_state
        audio_embeddings = torch.mean(audio_embeddings, dim=1)
        audio_features = self.audio_projection(audio_embeddings)
        
        text_outputs = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = self.text_projection(text_outputs.pooler_output)
        
        combined_features = torch.cat([audio_features, text_features], dim=1)
        fused_features = self.fusion(combined_features)
        fused_features = self.dropout(fused_features)
        
        return {
            'phase_logits': self.phase_classifier(fused_features),
            'filler_logits': self.filler_classifier(fused_features).squeeze(-1),
            'quality_logits': self.quality_regressor(fused_features).squeeze(-1),
            'enthusiasm_logits': self.enthusiasm_classifier(fused_features).squeeze(-1),
            'politeness_logits': self.politeness_classifier(fused_features).squeeze(-1)
        }

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        model_path = hf_hub_download(
            repo_id='alino-hcdc/calleval-wav2vec2-bert',
            filename='best_calleval_wav2vec2_bert_model.pth'
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = MultiModalCallEvalModel(
            wav2vec2_model_name="facebook/wav2vec2-base",
            bert_model_name="bert-base-uncased"
        )
        
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[10:] if key.startswith('_orig_mod.') else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        audio: Path = Input(description="Audio file of call segment"),
        text: str = Input(description="Transcript text of the audio segment")
    ) -> dict:
        
        waveform, sample_rate = torchaudio.load(str(audio))
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        waveform = waveform.flatten()
        max_length = 160000
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        elif len(waveform) < max_length:
            padding = max_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        audio_inputs = self.wav2vec2_processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        text_inputs = self.bert_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        audio_input_values = audio_inputs['input_values'].to(self.device)
        audio_attention_mask = torch.ones_like(audio_input_values)
        text_input_ids = text_inputs['input_ids'].to(self.device)
        text_attention_mask = text_inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                audio_input_values=audio_input_values,
                audio_attention_mask=audio_attention_mask,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask
            )
        
        phase_probs = torch.softmax(outputs['phase_logits'], dim=1).cpu().numpy()[0]
        phase_names = ['opening', 'middle', 'closing']
        
        return {
            'phase': {
                'predicted_phase': phase_names[int(phase_probs.argmax())],
                'probabilities': {
                    phase_names[i]: float(phase_probs[i]) 
                    for i in range(len(phase_names))
                }
            },
            'has_fillers': {
                'score': float(torch.sigmoid(outputs['filler_logits']).cpu().numpy()[0]),
                'prediction': 'yes' if torch.sigmoid(outputs['filler_logits']).cpu().numpy()[0] >= 0.5 else 'no'
            },
            'quality_score': float(outputs['quality_logits'].cpu().numpy()[0]),
            'enthusiasm': {
                'score': float(torch.sigmoid(outputs['enthusiasm_logits']).cpu().numpy()[0]),
                'prediction': 'enthusiastic' if torch.sigmoid(outputs['enthusiasm_logits']).cpu().numpy()[0] >= 0.5 else 'not_enthusiastic'
            },
            'politeness': {
                'score': float(torch.sigmoid(outputs['politeness_logits']).cpu().numpy()[0]),
                'prediction': 'polite' if torch.sigmoid(outputs['politeness_logits']).cpu().numpy()[0] >= 0.5 else 'not_polite'
            }
        }