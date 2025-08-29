import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Union, Tuple
from torch.distributions import Beta
from transformers import AutoModel, AutoTokenizer
from torch import Tensor

from subspaces import ridge_projector

logger = logging.getLogger(__name__)

def bound(a: Tensor, min: float, max: float) -> Tensor:
    return min + torch.sigmoid(a) * (max - min)

def make_mlp(in_dim: int, hidden1: int, hidden2: int, out_dim: int) -> nn.Module:
    """Factory function for MLP blocks with GELU + BatchNorm."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden1),
        nn.GELU(),
        nn.BatchNorm1d(hidden1),
        nn.Linear(hidden1, hidden2),
        nn.GELU(),
        nn.Linear(hidden2, out_dim)
    )

class TransformerSubspaceEmbedder(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        N: int,
        D: int,
        lbd: float,
        two_way: bool,
        cache_dir: str,
    ) -> None:
        """
        Initialize the TransformerSubspaceEmbedder.

        Args:
            base_model_name (str): Name of the HuggingFace transformer model.
            N (int): Number of vectors to span the subspace.
            D (int): Ambient space dimension.
            lbd (float): Ridge regularization parameter.
            two_way (bool): Whether to use two-way classification.
            cache_dir (str): Directory to cache pretrained models.
        """
        super().__init__()
        self.lbd = lbd # Ridge parameter
        self.D = D # Ambient space dimension
        self.N = N # Max subspace dimension
        self.two_way = two_way
        self.eps = 1e-6

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=cache_dir
        )
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            output_hidden_states=True
        )
        self.hidden_dim = self.base_model.config.hidden_size

        self.query = nn.Parameter(torch.randn(N, self.hidden_dim))
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.D),
        )

        self.alpha_ent = nn.Parameter(torch.tensor(-1.0))
        self.beta_ent = nn.Parameter(torch.tensor(-10.0))
        self.alpha_con = nn.Parameter(torch.tensor(-10.0))
        self.beta_con = nn.Parameter(torch.tensor(-1.0))

        if not self.two_way:
            self.mlp1a = make_mlp(self.D**2, 1024, 512, 512) 
            self.mlp1b = make_mlp(self.D**2, 1024, 512, 512) 
            self.mlp1c = make_mlp(self.D**2, 1024, 512, 512) 
            self.mlp1d = make_mlp(self.D**2, 1024, 512, 512) 
            self.mlp2 = nn.Sequential(nn.Linear(512*4, 1024), nn.GELU(), nn.Linear(1024, 1))

        self.w = nn.Parameter(torch.ones((1, 2 if self.two_way else 3)))
        self.b = nn.Parameter(torch.zeros((1, 2 if self.two_way else 3)))

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        """
        Forward pass: compute the smooth projection operator of the subspace.

        Args:
            input_ids (Tensor): Token IDs.
            attention_mask (Tensor): Attention mask for input tokens.
        
        Returns:
            Tensor: Projection operator representation of input.
        """
        x = self.forward_backbone(input_ids, attention_mask, **kwargs)
        proj = self.to_projection(x)
        return proj

    def forward_backbone(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        """
        Compute backbone transformer output and map it to intermediate features.

        Args:
            input_ids (Tensor): Token IDs.
            attention_mask (Tensor): Attention mask.
        
        Returns:
            Tensor: Batch of N D-dimensional vectors that span each subspace.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        layers = outputs.hidden_states[-3:]
        hidden_state = torch.cat(layers, dim=1)
        B = hidden_state.shape[0]
        # (B, N, base_model_dim)
        query = self.query.unsqueeze(0).expand(B, -1, -1)  
        # (B, N, base_model_dim)
        x, _ = self.attn(query=query, key=hidden_state, value=hidden_state)  
        # (B, N, D)
        x = self.mlp(x)
        return x
    
    def to_projection(self, x: Tensor) -> Tensor:
        """
        Compute smooth projection operator from vectors.

        Args:
            x (Tensor): Subspace vectors (B, N, D).
        
        Returns:
            Tensor: Subspace smooth projection operator (B, D, D).
        """
        P = ridge_projector(x.to(torch.float32), lbd=self.lbd)  # (B, D, D)
        return P.to(torch.float32)
        
    def classify(
        self,
        P_premise: Tensor,
        P_hypothesis: Tensor,
        targets: Optional[Tensor] = None,
        smoothing: float = 0.1,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Classify premise-hypothesis pairs using either 2-way or 3-way method.

        Args:
            P_premise (Tensor): Premise subspace smooth projector.
            P_hypothesis (Tensor): Hypothesis subspace smooth projector.
            targets (Optional[Tensor]): Ground-truth labels.
            smoothing (float): Label smoothing factor.
        
        Returns:
            Tensor or Tuple[Tensor, Tensor]: Logits or (logits, loss) if targets provided.
        """
        if self.two_way:
            return self.classify_2way(P_premise, P_hypothesis, targets, smoothing)
        else:
            return self.classify_3way(P_premise, P_hypothesis, targets, smoothing)
        
    def classify_3way(
        self,
        P_premise: Tensor,
        P_hypothesis: Tensor,
        targets: Optional[Tensor] = None,
        smoothing: float = 0.1,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        3-way classification (entailment, neutral, contradiction).

        Args:
            P_premise (Tensor): Premise smooth projector.
            P_hypothesis (Tensor): Hypothesis smooth projector.
            targets (Optional[Tensor]): Ground-truth labels.
            smoothing (float): Label smoothing factor.
        
        Returns:
            Tensor or Tuple[Tensor, Tensor]: Logits or (logits, loss).
        """
        prod = torch.bmm(P_premise, P_hypothesis.transpose(1,2))
        prod_inv = torch.bmm(P_hypothesis, P_premise.transpose(1,2))
        intersections = torch.einsum("bii->b", prod)
        
        prior_pre = torch.einsum("bii->b", P_premise) # (B,)
        p_hyp_pre = torch.clamp(intersections / (self.eps + prior_pre), self.eps, 1.0-self.eps)

        ent_dist = Beta(bound(self.alpha_ent, 1, 20), bound(self.beta_ent, 1, 20))
        con_dist = Beta(bound(self.alpha_con, 1, 20), bound(self.beta_con, 1, 20))

        neutral_logit = self.mlp2(torch.cat((self.mlp1a(P_premise.flatten(1,2)),
                                             self.mlp1b(P_hypothesis.flatten(1,2)),
                                             self.mlp1c(prod.flatten(1,2)),
                                             self.mlp1d(prod_inv.flatten(1,2))), dim=-1)).squeeze(1)

        log_1_minus_n = F.logsigmoid(-neutral_logit)
        s = torch.clamp(p_hyp_pre**2, self.eps, 1 - self.eps)

        logits_e = ent_dist.log_prob(s) + log_1_minus_n 
        logits_n = F.logsigmoid(neutral_logit) 
        logits_c = con_dist.log_prob(s) + log_1_minus_n
        logits = self.w * torch.stack((logits_e, logits_n, logits_c), dim=-1)  + self.b

        if targets is not None:
            loss = F.cross_entropy(logits, targets, label_smoothing=smoothing)
            return logits, loss   
        else:
            return logits   
        
    def classify_2way(
        self,
        P_premise: Tensor,
        P_hypothesis: Tensor,
        targets: Optional[Tensor] = None,
        smoothing: float = 0.1,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        2-way classification (entailment, non-entailment).

        Args:
            P_premise (Tensor): Premise smooth projector.
            P_hypothesis (Tensor): Hypothesis smooth projector.
            targets (Optional[Tensor]): Ground-truth labels.
            smoothing (float): Label smoothing factor.
        
        Returns:
            Tensor or Tuple[Tensor, Tensor]: Logits or (logits, loss).
        """
        prod = torch.bmm(P_premise, P_hypothesis.transpose(1,2))
        intersections = torch.einsum("bii->b", prod)
        
        prior_pre = torch.einsum("bii->b", P_premise) # (B,)
        p_hyp_pre = torch.clamp(intersections / (self.eps + prior_pre), self.eps, 1.0 - self.eps)

        # Priors
        ent_dist = Beta(bound(self.alpha_ent, 1, 20), bound(self.beta_ent, 1, 20))
        con_dist = Beta(bound(self.alpha_con, 1, 20), bound(self.beta_con, 1, 20))

        s = torch.clamp(p_hyp_pre**2, self.eps, 1 - self.eps)

        logits_e = ent_dist.log_prob(s)
        logits_c = con_dist.log_prob(s)
        logits = torch.stack((logits_e, logits_c), dim=-1)

        if targets is not None:
            loss = F.cross_entropy(logits, targets, label_smoothing=smoothing)
            return logits, loss   
        else:
            return logits  
        
    @torch.no_grad()
    def encode(
        self,
        text: List[str],
        max_length: int,
        device: Union[torch.device, str],
    ) -> Tensor:
        """
        Encode raw text into smooth subspace projector.

        Args:
            text (List[str]): List of input sentences.
            max_length (int): Maximum token length for truncation/padding.
            device (Union[torch.device, str]): Device to place tensors on.
        
        Returns:
            Tensor: Smooth projector representations of the input texts.
        """
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        for k in tokens:
            tokens[k] = tokens[k].to(device)
        P = self.forward(**tokens)
        return P
