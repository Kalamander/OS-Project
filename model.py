import json
import random
import numpy as np
import torch
import transformers
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import prompt,cot,got_step1,got_step2,got_step3

class VotingMachine:
    def __init__(self):
        self.votes = {}

    def vote(self, answer: str):
        self.votes[answer] = self.votes.get(answer, 0) + 1

    def get_results(self):
        if not self.votes:
            return ""
        return max(self.votes, key=self.votes.get)

class Model(nn.Module):
    def __init__(
            self,
            name: str,
            num_choices: int,
            mode: str = "rexgot"  # added mode with default
    ):
        super().__init__()

        self.name = name
        self.num_choices = num_choices
        self.mode = mode  # store the mode
        self.tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.model_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        self.max_length = 512
        self.hidden_size = 1024
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.classify = nn.Linear(self.hidden_size, 2)

    def score_input(self, content, labels, choices):
        device = next(self.parameters()).device

        if self.mode == "baseline":
            inputs = self.tokenizer_t5(
                content, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(device)

            labels_tokenized = self.tokenizer_t5(
                content, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).input_ids

            labels_tokenized[labels_tokenized == self.tokenizer_t5.pad_token_id] = -100

            out = self.model_t5(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels_tokenized.to(device)
            )
            return out.logits, out.loss

        outputs = []
        for text in content:
            context_1, got1 = got_step1(text)
            input_ids1 = self.tokenizer_t5(got1, return_tensors="pt").input_ids
            output1 = self.model_t5.generate(input_ids1.to(device))
            input_reconstructed = self.model_t5.generate(input_ids=output1)
            input_reconstructed_decoded = self.tokenizer_t5.decode(input_reconstructed[0], skip_special_tokens=True)
            input_ids_reconstructed = self.tokenizer_t5.encode(input_reconstructed_decoded, return_tensors="pt")
            if input_ids_reconstructed.shape[1] < input_ids1.shape[1]:
                input_ids_reconstructed = torch.nn.functional.pad(
                    input_ids_reconstructed,
                    (0, input_ids1.shape[1] - input_ids_reconstructed.shape[1]),
                    value=self.tokenizer_t5.pad_token_id
                )
            out1 = self.tokenizer_t5.decode(output1[0])
            out2 = []
            for i in range(len(choices)):
                context_2, got2 = got_step2(text, out1,choices[i])
                input_ids2 = self.tokenizer_t5(got2, return_tensors="pt").input_ids
                output2 = self.model_t5.generate(input_ids2.to(device))
                input_reconstructed2 = self.model_t5.generate(input_ids=output2)
                input_reconstructed_decoded2 = self.tokenizer_t5.decode(input_reconstructed2[0], skip_special_tokens=True)
                input_ids_reconstructed2 = self.tokenizer_t5.encode(input_reconstructed_decoded2, return_tensors="pt")
                if input_ids_reconstructed2.shape[1] < input_ids2.shape[1]:
                    input_ids_reconstructed2 = torch.nn.functional.pad(
                        input_ids_reconstructed2,
                        (0, input_ids2.shape[1] - input_ids_reconstructed2.shape[1]),
                        value=self.tokenizer_t5.pad_token_id
                    )
                out2.append(self.tokenizer_t5.decode(output2[0]))
            out2 = " ".join(out2)

            context_3, got3 = got_step3(text, out1, out2)
            num_answers = 3
            answers = []
            for _ in range(num_answers):
                input_ids3 = self.tokenizer_t5(got3, return_tensors="pt").input_ids
                output3 = self.model_t5.generate(input_ids3.to(device))
                input_reconstructed3 = self.model_t5.generate(input_ids=output3)
                input_reconstructed_decoded3 = self.tokenizer_t5.decode(input_reconstructed3[0],skip_special_tokens=True)
                input_ids_reconstructed3 = self.tokenizer_t5.encode(input_reconstructed_decoded3, return_tensors="pt")
                if input_ids_reconstructed3.shape[1] < input_ids3.shape[1]:
                    input_ids_reconstructed3 = torch.nn.functional.pad(
                        input_ids_reconstructed3,
                        (0, input_ids3.shape[1] - input_ids_reconstructed3.shape[1]),
                        value=self.tokenizer_t5.pad_token_id
                    )
                answer = self.tokenizer_t5.decode(output3[0], skip_special_tokens=True)
                answers.append(answer)
            voting_machine = VotingMachine()
            for answer in answers:
                voting_machine.vote(answer)
            out3 = voting_machine.get_results()
            outputs.append(out3)

        batch = self.tokenizer_t5(
            list(outputs),  # force list just in case
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        if len(batch["input_ids"].shape) == 1:
            batch["input_ids"] = batch["input_ids"].unsqueeze(0)
            batch["attention_mask"] = batch["attention_mask"].unsqueeze(0)

        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)

        labels_tokenized = self.tokenizer_t5(
            list(outputs),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids

        labels_tokenized[labels_tokenized == self.tokenizer_t5.pad_token_id] = -100

        out = self.model_t5(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=labels_tokenized.to(device)
        )

        return out["logits"], out.loss

    def forward(self, batch):
        content, labels, answer = batch
        device = next(self.parameters()).device
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        logits, loss = self.score_input(content, labels, answer)
        preds_cls = list(torch.argmax(logits, 1).cpu().numpy())
        positive_logits = logits[:, 1]
        preds = torch.argmax(positive_logits.reshape(-1, self.num_choices), 1)
        preds = list(preds.cpu().numpy())

        return loss, preds, preds_cls
