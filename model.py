from flax import linen as nn
from jax import Array, random
from dataclasses import dataclass
from jax import numpy as jnp, ensure_compile_time_eval

@dataclass
class PhiConfig:
    n_head: int = 32
    n_layer: int = 24
    n_embed: int = 2048
    rotary_dim: int = 32
    ln_eps: float = 1e-05
    n_positions: int = 2048
    vocab_size: int = 51200
    param_dtype: jnp.dtype = jnp.float16

def compute_cos_sin(config: PhiConfig) -> (Array, Array):
    t = jnp.arange(config.n_positions, dtype=jnp.float32)
    inv_freq = 1 / (10000 ** (jnp.arange(0, config.rotary_dim, 2, dtype=jnp.float32) / config.rotary_dim))
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs).astype(config.param_dtype), jnp.sin(freqs).astype(config.param_dtype)

def apply_rotary_emb(qkv: Array, cos: Array, sin: Array) -> Array:
    _, seq_len, _, _, _ = qkv.shape
    _, rotary_dim = cos.shape
    rotary_dim *= 2

    q_rot = qkv[:, :, 0, :, :rotary_dim]
    q_pass = qkv[:, :, 0, :, rotary_dim:]

    k_rot = qkv[:, :, 1, :, :rotary_dim]
    k_pass = qkv[:, :, 1, :, rotary_dim:]

    q1, q2 = jnp.split(q_rot.astype(jnp.float32), 2, axis=-1)
    k1, k2 = jnp.split(k_rot.astype(jnp.float32), 2, axis=-1)
    c, s = cos[:seq_len][:, None, :].astype(jnp.float32), sin[:seq_len][:, None, :].astype(jnp.float32)

    q_rot = jnp.concatenate([q1 * c - q2 * s, q1 * s + q2 * c], axis=-1).astype(qkv.dtype)
    k_rot = jnp.concatenate([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).astype(qkv.dtype)

    return jnp.concatenate([
        jnp.concatenate([q_rot, q_pass], axis=-1)[:, :, None, :, :],
        jnp.concatenate([k_rot, k_pass], axis=-1)[:, :, None, :, :],
        qkv[:, :, 2:3, :, :]
    ], axis=2)

class SelfAttention(nn.Module):
    config: PhiConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        batch_size, seq_len, n_embed = x.shape

        with ensure_compile_time_eval():
            cos, sin = compute_cos_sin(self.config)

        scale = (n_embed // self.config.n_head) ** -0.5
        mask = jnp.triu(jnp.full((seq_len, seq_len), -10000.0, dtype=jnp.float16), 1)
        qkv = nn.Dense(features=3 * self.config.n_embed, use_bias=True, param_dtype=self.config.param_dtype)(x)
        qkv = jnp.reshape(qkv, (batch_size, seq_len, 3, self.config.n_head, n_embed // self.config.n_head))
        qkv = apply_rotary_emb(qkv, cos, sin)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = jnp.split(qkv, 3, axis=0)
        a = (q @ jnp.swapaxes(k, -2, -1)) * scale + mask
        a = nn.softmax(a, axis=-1)
        a = (a @ v).swapaxes(1, 2).reshape(batch_size, seq_len, n_embed)
        return nn.Dense(features=n_embed, use_bias=True, param_dtype=self.config.param_dtype)(a)

class MLP(nn.Module):
    config: PhiConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        h = nn.Dense(features=self.config.n_embed * 4, use_bias=True, param_dtype=self.config.param_dtype)(x)
        return nn.Dense(features=self.config.n_embed, use_bias=True, param_dtype=self.config.param_dtype)(nn.gelu(h))

class Block(nn.Module):
    config: PhiConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        h = nn.LayerNorm(epsilon=self.config.ln_eps, param_dtype=self.config.param_dtype)(x)
        a = SelfAttention(self.config)(h)
        return a + MLP(self.config)(h)

class Phi(nn.Module):
    config: PhiConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        h = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.n_embed, param_dtype=self.config.param_dtype)(x)
        for _ in range(self.config.n_layer):
            h = Block(self.config)(h)
        h = nn.LayerNorm(epsilon=self.config.ln_eps, param_dtype=self.config.param_dtype)(h)
        return nn.Dense(self.config.vocab_size, use_bias=True, param_dtype=self.config.param_dtype)(h)
    
phi = Phi(PhiConfig())
x = random.randint(random.PRNGKey(0), (128, 2048), 0, 51200, dtype=jnp.int32)
phi.tabulate(random.PRNGKey(0), x)