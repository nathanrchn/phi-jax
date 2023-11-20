from jax import numpy as jnp, config
from flax import linen as nn
from jax import Array, random

def apply_rotary_emb(x: Array) -> Array:
    # rotary_dim = 32
    return x

class MHA(nn.Module):
    n_heads: int = 16
    head_dim: int = 64
    n_embed: int = 10 # 1024

    @nn.compact
    def __call__(self, x: Array) -> Array:
        qkv = nn.Dense(features=3 * self.n_heads * self.head_dim, use_bias=True)(x)
        batch_size, seq_len, _ = qkv.shape
        mask = jnp.triu(jnp.full((seq_len, seq_len), -10000.0, dtype=jnp.float32), 1)
        qkv = apply_rotary_emb(qkv)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        s = nn.dot_product_attention(q, k, v, mask=mask)
        a = nn.softmax(s, axis=-1)
        return nn.Dense(features=self.n_embed, use_bias=True)(a)
    
class MLP(nn.Module):
    n_inner: int = 40 # 4096

    @nn.compact
    def __call__(self, x: Array) -> Array:
        h = nn.Dense(features=self.n_inner, use_bias=True)(x)
        return nn.Dense(features=x.shape[-1], use_bias=True)(nn.gelu(h))
    
class Block(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        h = nn.LayerNorm(epsilon=1e-5)(x)
        return MHA()(h) + MLP()(h)
    
class Phi(nn.Module):
    n_layer: int = 20
    n_embd: int = 10 # 1024
    vocab_size: int = 2 # 50304

    @nn.compact
    def __call__(self, x: Array) -> Array:
        h = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embd)(x)
        for _ in range(self.n_layer):
            h = Block()(h)
        h = nn.LayerNorm(epsilon=1e-5)(h)
        return nn.Dense(self.vocab_size, use_bias=True, dtype=jnp.float32)(h)
    
phi = Phi()
x = random.randint(random.PRNGKey(0), (1, 10), 0, 2) # random.randint(random.PRNGKey(0), (1, 1024), 0, 50304)
variables = phi.init(random.PRNGKey(0), x)

stack = [(k, v) for k, v in variables["params"].items()]
stack.reverse()

while len(stack) > 0:
    k, v  = stack.pop()

    if isinstance(v, dict):
        stack += [(k + "." + k_, v_) for k_, v_ in v.items()]
    else:
        print(k)

print(phi.apply(variables, x))