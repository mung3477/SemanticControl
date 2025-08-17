
import math
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate
from einops import rearrange


def _calc_head_by_head_mean_ratio(attn_weight: torch.Tensor, attn_bias: torch.Tensor):
	reshaped_weight = attn_weight.reshape((*attn_weight.shape[:2], -1))
	attn_weight_mean_per_head = reshaped_weight.sum(axis=-1) / (reshaped_weight > 0).sum(axis=-1).float()

	reshaped_bias = attn_bias.reshape((*attn_bias.shape[:2], -1))
	attn_bias_mean_per_head = reshaped_bias.sum(axis=-1) / (reshaped_bias > 0).sum(axis=-1).float()

	scaler = attn_weight_mean_per_head / attn_bias_mean_per_head.to(attn_weight.device)
	return scaler.reshape(*scaler.shape, 1, 1)

def _calc_mean_ratio(attn_weight: torch.Tensor, attn_bias: torch.Tensor):
	scaler = attn_weight[attn_bias > 0].mean(axis=-1) / attn_bias[attn_bias > 0].mean(axis=-1)
	return scaler

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
	# Efficient implementation equivalent to the following:
	L, S = query.size(-2), key.size(-2)
	scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
	attn_bias = torch.zeros(L, S, dtype=query.dtype)
	if is_causal:
		assert attn_mask is None
		temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
		attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
		attn_bias.to(query.dtype)

	if attn_mask is not None:
		if attn_mask.dtype == torch.bool:
			attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
		else:
			# attn_bias += attn_mask
			attn_bias = attn_mask
	attn_weight = query @ key.transpose(-2, -1) * scale_factor
	attn_weight = torch.softmax(attn_weight, dim=-1)

	if attn_bias is not None and attn_bias.sum() > 0:
		scaler = _calc_mean_ratio(attn_weight, attn_bias)
		attn_weight += 3 * scaler * attn_bias.to(attn_weight.device)

	return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight

def attn_call(
		self,
		attn: Attention,
		hidden_states: torch.Tensor,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		temb: Optional[torch.Tensor] = None,
		height: int = None,
		width: int = None,
		timestep: Optional[torch.Tensor] = None,
		*args,
		**kwargs,
	) -> torch.Tensor:
		r"""
			To save attention score as an image, height parameter should be added.
		"""
		if len(args) > 0 or kwargs.get("scale", None) is not None:
			deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
			deprecate("scale", "1.0.0", deprecation_message)

		residual = hidden_states

		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)

		input_ndim = hidden_states.ndim

		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

		batch_size, sequence_length, _ = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)
		attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		query = attn.head_to_batch_dim(query)
		key = attn.head_to_batch_dim(key)
		value = attn.head_to_batch_dim(value)

		attention_probs = attn.get_attention_scores(query, key, attention_mask)
		####################################################################################################
		# attention score should be saved as an image
		# (batch * num_heads, seq_len, head_dim)
		if hasattr(self, "store_attn_map"):
			self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height)
			self.timestep = int(timestep.item())
		####################################################################################################
		hidden_states = torch.bmm(attention_probs, value)
		hidden_states = attn.batch_to_head_dim(hidden_states)

		# linear proj
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)

		if input_ndim == 4:
			hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

		if attn.residual_connection:
			hidden_states = hidden_states + residual

		hidden_states = hidden_states / attn.rescale_output_factor

		return hidden_states

def attn_call2_0(
	self,
	attn: Attention,
	hidden_states: torch.Tensor,
	encoder_hidden_states: Optional[torch.Tensor] = None,
	attention_mask: Optional[torch.Tensor] = None,
	temb: Optional[torch.Tensor] = None,
	height: int = None,
	width: int = None,
	timestep: Optional[torch.Tensor] = None,
	*args,
	**kwargs,
) -> torch.Tensor:
	if len(args) > 0 or kwargs.get("scale", None) is not None:
		deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
		deprecate("scale", "1.0.0", deprecation_message)

	residual = hidden_states
	if attn.spatial_norm is not None:
		hidden_states = attn.spatial_norm(hidden_states, temb)

	input_ndim = hidden_states.ndim

	if input_ndim == 4:
		batch_size, channel, height, width = hidden_states.shape
		hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

	batch_size, sequence_length, _ = (
		hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
	)

	if attention_mask is not None:
		attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
		# scaled_dot_product_attention expects attention_mask shape to be
		# (batch, heads, source_length, target_length)
		attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
	################################################
	elif "attn_force_range" in kwargs and "attn_bias" in kwargs:
		attn_force_range = kwargs["attn_force_range"]
		attn_bias = kwargs["attn_bias"]
		attn_bias = attn_bias / len(attn_force_range)
		attention_mask = torch.zeros((batch_size, attn.heads, height * width, sequence_length))
		attention_mask[:, :, :, attn_force_range[0]:attn_force_range[-1] + 1] = torch.stack([attn_bias] * len(attn_force_range), axis=-1)
	################################################

	if attn.group_norm is not None:
		hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

	query = attn.to_q(hidden_states)

	if encoder_hidden_states is None:
		encoder_hidden_states = hidden_states
	elif attn.norm_cross:
		encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

	key = attn.to_k(encoder_hidden_states)
	value = attn.to_v(encoder_hidden_states)

	inner_dim = key.shape[-1]
	head_dim = inner_dim // attn.heads

	query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

	key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
	value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

	# the output of sdp = (batch, num_heads, seq_len, head_dim)
	# TODO: add support for attn.scale when we move to Torch 2.1
	####################################################################################################
	if hasattr(self, "store_attn_map"):
		hidden_states, attention_probs = scaled_dot_product_attention(
			query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
		)

		if "attn_preserve_range" in kwargs:
			preserve_range = kwargs["attn_preserve_range"]
			self.attn_bias = attention_probs[:, :, :, preserve_range[0]:preserve_range[-1] + 1].mean(axis = -1)		# batch attn_head (h w)


		if "token_last_idx" in kwargs:
			""" re-weight the attention values by ignoring the attention of ⟨𝑠𝑜𝑡⟩ and <eot> """
			attention_probs = attention_probs[:, :, :, 1:kwargs["token_last_idx"]]
			attention_probs *= 1000
			attention_probs = torch.nn.functional.softmax(attention_probs, dim=-1)

		attention_probs = rearrange(attention_probs, 'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ', h=height) # detach height*width
		self.attn_map = attention_probs
		self.timestep = int(timestep.item())

	else:
		hidden_states = F.scaled_dot_product_attention(
			query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
		)
	####################################################################################################

	hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # (b,attn_head,h*w,attn_dim) -> (b,h*w,attn_head*attn_dim)
	hidden_states = hidden_states.to(query.dtype)

	# linear proj
	hidden_states = attn.to_out[0](hidden_states)
	# dropout
	hidden_states = attn.to_out[1](hidden_states)

	if input_ndim == 4:
		hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

	if attn.residual_connection:
		hidden_states = hidden_states + residual

	hidden_states = hidden_states / attn.rescale_output_factor

	return hidden_states
