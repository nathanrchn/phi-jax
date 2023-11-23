from flax import linen as nn
from flax import traverse_util
from dataclasses import dataclass
from jax import Array, jit, random
from flax.training.train_state import TrainState
from optax import adamw, set_to_zero, multi_transform, l2_loss
from jax import numpy as jnp, value_and_grad, ensure_compile_time_eval

@dataclass
class PhiConfig:
    n_head: int = 32
    n_layer: int = 24
    n_embed: int = 2048
    rotary_dim: int = 32
    ln_eps: float = 1e-05
    n_positions: int = 2048
    vocab_size: int = 51200
    target_hidden_size: int = 2048
    param_dtype: jnp.dtype = jnp.bfloat16

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
    def __call__(self, x: Array) -> (Array, Array):
        h = nn.Dense(features=self.config.n_embed * 4, use_bias=True, param_dtype=self.config.param_dtype)(x)
        h = nn.Dense(features=self.config.n_embed, use_bias=True, param_dtype=self.config.param_dtype)(nn.gelu(h))

        l = nn.Dense(features=self.config.target_hidden_size, use_bias=True, param_dtype=self.config.param_dtype)(x)
        l = nn.Dense(features=self.config.n_embed, use_bias=True, param_dtype=self.config.param_dtype)(nn.gelu(l))

        return (h, l2_loss(l, h))

class Block(nn.Module):
    config: PhiConfig

    @nn.compact
    def __call__(self, x: Array) -> (Array, Array):
        h = nn.LayerNorm(epsilon=self.config.ln_eps, param_dtype=self.config.param_dtype)(x)
        a = SelfAttention(self.config)(h)
        (h, loss) = MLP(self.config)(h)
        return (a + h, loss)

class Phi(nn.Module):
    config: PhiConfig

    @nn.compact
    def __call__(self, x: Array) -> list[Array]:
        losses = jnp.zeros((1,), dtype=jnp.float16)
        h = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.n_embed, param_dtype=self.config.param_dtype)(x)
        for _ in range(self.config.n_layer):
            (h, loss) = Block(self.config)(h)
            losses += loss
        # useless layers while training
        # h = nn.LayerNorm(epsilon=self.config.ln_eps, param_dtype=self.config.param_dtype)(h)
        # o = nn.Dense(self.config.vocab_size, use_bias=True, param_dtype=self.config.param_dtype)(h)
        return losses
    
# light config
config = PhiConfig(n_head=4, n_layer=2, n_embed=256, rotary_dim=16, n_positions=16, vocab_size=51200, target_hidden_size=16)

# config = PhiConfig()
phi = Phi(config)
x = random.randint(random.PRNGKey(0), (1, config.n_positions), 0, 51200, dtype=jnp.int32)
# variables = phi.init(random.PRNGKey(0), x)
print(phi.tabulate(random.PRNGKey(0), x))

# variables = phi.init(random.PRNGKey(0), jnp.ones((4, config.n_positions), dtype=jnp.int32))

# print(traverse_util.path_aware_map(lambda k, v: print(k, v.shape), variables["params"]))

# def init_train_state(batch_size) -> TrainState:
#     config = PhiConfig()
#     phi = Phi(config)
#     model = torch.load("pytorch_model.bin")
#     variables = phi.init(random.PRNGKey(0), jnp.ones((batch_size, config.n_positions), dtype=jnp.int32))
#     variables = load_model_into_flax(model, variables)

#     partition_optimizers = {"trainable": adamw(0.001), "frozen": set_to_zero()}
#     param_partitions = traverse_util.path_aware_map(
#         lambda path, _: "trainable" if ("Dense_2" in path or  "Dense_3" in path) else "frozen", variables["params"])

#     return TrainState.create(
#         apply_fn=phi.apply,
#         tx=multi_transform(partition_optimizers, param_partitions),
#         params=variables["params"]
#     )

# model = load("pytorch_model.bin")

# def load_model_into_flax(model, variables) -> dict:
#     j = 0
#     for param_name in model:
#         param_name = param_name.replace("layers.", "")
#         i = int(param_name.split(".")[0])
#         jnp_array = jnp.array(model[param_name].numpy()).astype(jnp.float16)
#         if i == 0:
#             variables["params"][f"Embed_{i}"]["embedding"] = jnp_array
#         elif not param_name.endswith("inv_freq"):
#             match i:
#                 case 0: variables["params"][f"Block_{i}"]["LayerNorm_0"]["scale"] = jnp_array; j += 1
#                 case 1: variables["params"][f"Block_{i}"]["LayerNorm_0"]["bias"] = jnp_array; j += 1
#                 case 2: variables["params"][f"Block_{i}"]["SelfAttention_0"]["Dense_0"]["kernel"] = jnp_array; j += 1
#                 case 3: variables["params"][f"Block_{i}"]["SelfAttention_0"]["Dense_0"]["bias"] = jnp_array; j += 1
#                 case 4: variables["params"][f"Block_{i}"]["SelfAttention_0"]["Dense_1"]["kernel"] = jnp_array; j += 1
#                 case 5: variables["params"][f"Block_{i}"]["SelfAttention_0"]["Dense_1"]["bias"] = jnp_array; j += 1
#                 case 6: variables["params"][f"Block_{i}"]["MLP_0"]["Dense_0"]["kernel"] = jnp_array; j += 1
#                 case 7: variables["params"][f"Block_{i}"]["MLP_0"]["Dense_0"]["bias"] = jnp_array; j += 1
#                 case 8: variables["params"][f"Block_{i}"]["MLP_0"]["Dense_1"]["kernel"] = jnp_array; j += 1
#                 case 9: variables["params"][f"Block_{i}"]["MLP_0"]["Dense_1"]["bias"] = jnp_array; j += 1
#                 case 10: j = 0
#     return variables

# @jit
# def train_step(state: TrainState, batch: Array):
#     def loss_fn(params):
#         return phi.apply({"params": params}, batch)
    
#     grad_fn = value_and_grad(loss_fn)
#     losses, grads = grad_fn(state.params)
#     return state.apply_gradients(grads=grads)